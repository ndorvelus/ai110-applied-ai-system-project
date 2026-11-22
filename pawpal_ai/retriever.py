"""Retrieval layer: BM25 search over pet-care guides plus the owner's own pet records.

Two sources feed the retriever:

* ``load_guides`` - the markdown guides in ``knowledge/``, split into one chunk per
  ``##`` section. Each guide declares the species it applies to in front matter, and
  sections may carry ``TASK:`` lines the agent turns into schedule tasks.
* ``pet_record_passages`` - the owner's live data (pets, ages, existing tasks), turned
  into short text records so questions like "what does Biscuit have today?" are
  grounded in the user's data rather than in generic advice.

Scoring is plain BM25 with a species filter, implemented here so the project has no
heavy dependencies and every score is explainable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from pawpal_system import Owner

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "my", "of", "on",
    "or", "should", "so", "that", "the", "their", "them", "there", "this", "to", "up",
    "was", "what", "when", "which", "who", "why", "will", "with", "you", "your", "me",
    "much", "many", "need", "needs", "day", "today", "about", "any", "get", "give",
    "plan", "make", "please", "help", "minute", "minutes", "year", "old", "just",
}

# Light query expansion so everyday words reach the guides' vocabulary.
SYNONYMS = {
    "walk": ["exercise"],
    "food": ["feeding", "meal"],
    "feed": ["feeding", "meal"],
    "eat": ["feeding", "meal"],
    "ate": ["toxic", "poisoning"],
    "poison": ["toxic", "poisoning"],
    "brush": ["brushing", "grooming"],
    "teeth": ["dental", "tooth", "toothpaste"],
    "med": ["medication", "dose"],
    "medicine": ["medication", "dose"],
    "pill": ["medication", "dose"],
    "senior": ["senior"],
    "elderly": ["senior"],
    "puppy": ["puppies"],
    "kitten": ["kitten"],
    "litter": ["box", "scoop"],
    "play": ["enrichment", "toy"],
    "bored": ["enrichment", "mental", "boredom"],
    "hot": ["heat", "weather"],
    "hay": ["diet"],
    "benadryl": ["medication", "dose"],
    "ibuprofen": ["medication", "toxic"],
    "tylenol": ["medication", "toxic"],
    "aspirin": ["medication", "toxic"],
}


def _stem(word: str) -> str:
    """Very small stemmer: strip a trailing plural 's' from longer words."""
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed and plurals folded."""
    return [
        _stem(w) for w in re.findall(r"[a-z]+", text.lower())
        if w not in STOPWORDS and len(w) > 1
    ]


def expand_query(tokens: list[str]) -> list[str]:
    """Add synonyms for known everyday words (each original token is kept)."""
    expanded = list(tokens)
    for t in tokens:
        expanded.extend(tokenize(" ".join(SYNONYMS.get(t, []))))
    return expanded


@dataclass
class Passage:
    """One retrievable chunk of text and where it came from."""

    source: str  # e.g. "dog_exercise.md#Senior dogs" or "pet_record:Biscuit"
    title: str
    text: str
    species: set[str] = field(default_factory=set)  # empty means applies to all
    task_lines: list[str] = field(default_factory=list)
    score: float = 0.0
    kind: str = "guide"  # "guide" or "pet_record"


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split a ``---`` front-matter block off a markdown document."""
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def load_guides(directory: Path = KNOWLEDGE_DIR) -> list[Passage]:
    """Load every markdown guide and split it into one Passage per ``##`` section."""
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text())
        species = {s.strip() for s in meta.get("species", "").split(",") if s.strip()}
        doc_title = next(
            (ln[2:].strip() for ln in body.splitlines() if ln.startswith("# ")), path.stem
        )
        for section in re.split(r"\n(?=## )", body):
            lines = section.strip().splitlines()
            if not lines or not lines[0].startswith("## "):
                continue
            heading = lines[0][3:].strip()
            task_lines = [ln[5:].strip() for ln in lines if ln.startswith("TASK:")]
            text = " ".join(ln for ln in lines[1:] if not ln.startswith("TASK:")).strip()
            passages.append(
                Passage(
                    source=f"{path.name}#{heading}",
                    title=f"{doc_title}: {heading}",
                    text=text,
                    species=species,
                    task_lines=task_lines,
                )
            )
    return passages


def pet_record_passages(owner: Owner | None) -> list[Passage]:
    """Turn the owner's pets and existing tasks into retrievable text records."""
    if owner is None:
        return []
    records = []
    for pet in owner.pets:
        pending = [t for t in pet.tasks if not t.completed]
        task_text = "; ".join(
            f"{t.title} ({t.category}, {t.duration_minutes} min, {t.priority} priority"
            f"{', at ' + t.time if t.time else ''})"
            for t in pending
        ) or "no pending tasks"
        records.append(
            Passage(
                source=f"pet_record:{pet.name}",
                title=f"{pet.name}'s record",
                text=(
                    f"{pet.name} is a {pet.age}-year-old {pet.breed} {pet.species}. "
                    f"Owner {owner.name} has {owner.available_minutes} minutes available "
                    f"today. Current pending tasks: {task_text}."
                ),
                species={pet.species},
                kind="pet_record",
            )
        )
    return records


class Retriever:
    """BM25 retriever over guide passages plus, optionally, the owner's pet records.

    Args:
        guides: Passages from the knowledge base (loaded from disk by default).
        use_pet_records: Include the owner's pet records as a second source.
    """

    k1 = 1.5
    b = 0.75

    def __init__(self, guides: list[Passage] | None = None, use_pet_records: bool = True):
        self.guides = guides if guides is not None else load_guides()
        self.use_pet_records = use_pet_records

    def task_sections(self, species: str) -> list[Passage]:
        """Custom index: every guide section that carries TASK lines for a species.

        Used by the agent's planner so a schedule can draw on all relevant care
        routines, not only the few sections that happen to match the request words.
        """
        return [
            p for p in self.guides
            if p.task_lines and (not p.species or species in p.species)
        ]

    def search(
        self,
        query: str,
        owner: Owner | None = None,
        species: str | None = None,
        pet_names: list[str] | None = None,
        top_k: int = 4,
    ) -> list[Passage]:
        """Return up to top_k passages for a query, best first, each with a ``score``.

        Passages tagged for a different species are dropped. A pet record named in the
        query gets a fixed boost so the user's own data is not drowned out by guides.
        """
        records = pet_record_passages(owner) if self.use_pet_records else []
        corpus = [
            p for p in self.guides + records
            if species is None or not p.species or species in p.species
        ]
        if not corpus:
            return []
        docs = [tokenize(p.title + " " + p.text) for p in corpus]
        avg_len = sum(map(len, docs)) / len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)

        q_terms = set(expand_query(tokenize(query)))
        scored = []
        for passage, doc in zip(corpus, docs):
            tf = Counter(doc)
            score = 0.0
            for term in q_terms & tf.keys():
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                freq = tf[term]
                score += idf * freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * len(doc) / avg_len)
                )
            if passage.kind == "pet_record" and passage.source.split(":", 1)[1] in (
                pet_names or []
            ):
                score += 2.0
            if score > 0:
                scored.append((score, passage))

        scored.sort(key=lambda sp: sp[0], reverse=True)
        return [replace(p, score=round(s, 3)) for s, p in scored[:top_k]]
