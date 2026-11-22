"""Answer writers: turn retrieved passages into a reply, in one of two styles.

Two backends share one interface (``write(question, passages, style, dosage_mode)``):

* ``OfflineWriter`` (default) - deterministic and extractive: it picks the sentences
  from the retrieved passages that best match the question. Needs no API key, so the
  whole system, its tests and its evaluation are reproducible on any machine.
* ``ClaudeWriter`` - calls Claude through the Anthropic SDK with the same passages as
  context. Enabled with ``PAWPAL_LLM=claude`` when the ``anthropic`` package and
  credentials are available; any API failure falls back to the offline writer.

Two styles show the specialization stretch feature:

* ``baseline`` - a plain answer, no structure or citations.
* ``pawpal`` - the specialized "PawPal care coach" style, constrained by few-shot
  examples: at most 3 bullets, each ending in a [n] citation, a "Next step" line, and a
  vet-safety note. ``eval/run_eval.py`` measures how the two styles differ.
"""

from __future__ import annotations

import logging
import os
import re

from .retriever import Passage, expand_query, tokenize

log = logging.getLogger("pawpal_ai")

CLAUDE_MODEL = "claude-opus-5-5"

SAFETY_NOTE = (
    "Safety: this is general care guidance, not a diagnosis. Ask your vet about "
    "anything medical."
)

# Few-shot examples that define the specialized style (used verbatim in the Claude
# prompt, and mirrored by the offline formatter).
FEW_SHOT = [
    (
        "How often should I feed my adult cat?",
        "- Adult cats do well on two or more small measured meals a day [1].\n"
        "- Measure portions, because indoor cats gain weight easily [1].\n"
        "Next step: add 'Feed breakfast' and 'Feed dinner' as daily high-priority tasks.\n"
        + SAFETY_NOTE,
    ),
    (
        "Is it okay to walk my dog when it's really hot?",
        "- Walk early in the morning or late in the evening on hot days [1].\n"
        "- If pavement is too hot for the back of your hand for seven seconds, it is too "
        "hot for paws [1].\n"
        "Next step: move today's walk to before 9:00 and bring water.\n" + SAFETY_NOTE,
    ),
]

STYLE_RULES = (
    "You are PawPal, a friendly pet-care coach. Answer ONLY from the numbered context "
    "passages. Format: at most 3 bullet points starting with '- ', each ending with the "
    "citation of the passage it came from, like [1]. Then one line starting with "
    "'Next step:' that gives one concrete action. Then this exact line: "
    f"'{SAFETY_NOTE}'. Stay under 120 words. If the context does not answer the "
    "question, say so in one bullet and suggest asking a vet. Never give medication "
    "doses or numbers of milligrams."
)

DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|milligrams?|ml|tablets?)\b", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def _ranked_sentences(question: str, passages: list[Passage]) -> list[tuple[int, str]]:
    """(passage_number, sentence) pairs, best match to the question first."""
    q = set(expand_query(tokenize(question)))
    scored = []
    for n, p in enumerate(passages, start=1):
        if p.kind == "pet_record":
            continue
        for order, sentence in enumerate(_sentences(p.text)):
            overlap = len(q & set(tokenize(sentence)))
            # Prefer overlap, then better-ranked passages, then earlier sentences.
            scored.append((overlap, -n, -order, n, sentence))
    scored.sort(reverse=True)
    return [(n, s) for overlap, _, _, n, s in scored if overlap > 0]


def strip_doses(text: str) -> str:
    """Replace any dose-like number ("5 mg", "2 tablets") with a pointer to the vet."""
    return DOSE_RE.sub("[dose: ask your vet]", text)


class OfflineWriter:
    """Deterministic extractive writer used by default and in all tests."""

    name = "offline"

    def write(
        self, question: str, passages: list[Passage], style: str = "pawpal",
        dosage_mode: bool = False, next_step: str | None = None,
    ) -> str:
        ranked = _ranked_sentences(question, passages)
        if style == "baseline":
            if not ranked:
                return "I'm not sure about that."
            return " ".join(s for _, s in ranked[:3])

        body: list[str] = []
        if dosage_mode:
            body.append("- I can't recommend a dose; only your vet can set one.")
        if not ranked:
            body.append(
                "- I couldn't find this in my pet-care guides, so I won't guess; "
                "please ask your vet."
            )
        seen = set()
        for n, sentence in ranked:
            if len(body) == 3:
                break
            if sentence not in seen:
                seen.add(sentence)
                body.append(f"- {sentence.rstrip('.')} [{n}].")
        lines = body + [f"Next step: {next_step or 'add the matching task to your schedule.'}"]
        lines.append(SAFETY_NOTE)
        return strip_doses("\n".join(lines))


class ClaudeWriter:
    """Writes answers with Claude, using the retrieved passages as grounding context."""

    name = "claude"

    def __init__(self) -> None:
        import anthropic  # imported lazily so the offline path needs no SDK

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.fallback = OfflineWriter()

    def write(
        self, question: str, passages: list[Passage], style: str = "pawpal",
        dosage_mode: bool = False, next_step: str | None = None,
    ) -> str:
        context = "\n".join(
            f"[{n}] ({p.source}) {p.text}" for n, p in enumerate(passages, start=1)
        )
        messages: list[dict] = []
        if style == "baseline":
            system = "Answer the user's pet-care question using the context provided."
        else:
            system = STYLE_RULES
            for q, a in FEW_SHOT:
                messages += [
                    {"role": "user", "content": f"Context:\n[1] (example)\n\nQuestion: {q}"},
                    {"role": "assistant", "content": a},
                ]
        hint = f"\nSuggested next step: {next_step}" if next_step else ""
        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}{hint}"}
        )
        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                output_config={"effort": "low"},
                system=system,
                messages=messages,
            )
        except self._anthropic.APIError as exc:
            log.warning("Claude call failed (%s); using offline writer", exc)
            return self.fallback.write(question, passages, style, dosage_mode, next_step)
        if response.stop_reason == "refusal":
            log.warning("Claude refused; using offline writer")
            return self.fallback.write(question, passages, style, dosage_mode, next_step)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return strip_doses(text) if dosage_mode else text


def get_writer() -> OfflineWriter | ClaudeWriter:
    """Pick the writer from ``PAWPAL_LLM`` (``offline`` by default)."""
    if os.environ.get("PAWPAL_LLM", "offline").lower() == "claude":
        try:
            return ClaudeWriter()
        except Exception as exc:  # missing SDK or credentials
            log.warning("Claude writer unavailable (%s); using offline writer", exc)
    return OfflineWriter()
