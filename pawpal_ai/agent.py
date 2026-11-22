"""PawPal agent: plan -> retrieve -> act -> check -> revise -> answer.

The agent turns a free-text request ("Plan today for Biscuit" or "Can cats eat
lilies?") into either a grounded answer or a schedule built with the original PawPal+
``Schedule`` class. Every step is recorded in a trace so the reasoning can be read
afterwards (``AgentResult.trace``; also appended to ``logs/agent_traces.jsonl``).

Steps:

1. **guardrail** - ``check_input`` screens for empty/oversized input, prompt
   injection, emergencies (stop and send to a vet) and dose questions ("no doses" mode).
2. **plan** - decide the intent (``schedule`` or ``question``), which pets the request
   is about, and the ordered list of tools to call.
3. **retrieve** - BM25 over the knowledge guides plus the owner's pet records
   (``Retriever.search``), once per target species so species filtering applies.
4. **propose_tasks** - turn ``TASK:`` lines in the retrieved guide sections into
   ``Task`` objects, filtered by the pet's age and skipping tasks the pet already has.
5. **build_schedule** - add the proposals to a *copy* of the owner and run the original
   ``Schedule.generate_plan`` plus ``find_next_available_slot`` for start times.
6. **critique / revise** - ``check_plan`` flags skipped high-priority tasks; the agent
   then drops its own lowest-value proposals and re-plans (at most ``MAX_REVISIONS``).
7. **answer** - the writer (offline or Claude) phrases the reply from the passages;
   ``check_answer`` verifies it is grounded and dose-free, rewriting once if not.
8. **confidence** - a 0-1 score from retrieval strength, critique result and checks.

The owner passed in is never modified; callers apply ``result.proposed_tasks`` if the
user accepts the plan (the Streamlit app does this with an "Add to my tasks" button).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from pawpal_system import Owner, Pet, Schedule, Task

from .guardrails import check_answer, check_input, check_plan
from .llm import OfflineWriter, get_writer, strip_doses
from .retriever import Passage, Retriever

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
MAX_REVISIONS = 2
FLEXIBLE_CATEGORIES = {"exercise", "enrichment"}
MIN_FLEX_MINUTES = 10  # never shorten an activity below this
STRONG_SCORE = 6.0  # BM25 score treated as a confident match when normalising

log = logging.getLogger("pawpal_ai")

SCHEDULE_WORDS = re.compile(
    r"\b(plan|schedule|routine|set up|organi[sz]e|tasks for)\b"
)
# Preferred start-time windows by title keyword; anything else goes 07:00-21:00.
TIME_WINDOWS = [
    (r"breakfast|morning", ("07:00", "10:00")),
    (r"dinner|evening", ("17:00", "21:00")),
    (r"walk|roam|play|training", ("07:00", "19:00")),
]
SPECIES_WORDS = {
    "dog": r"\b(dog|dogs|puppy|puppies|pup)\b",
    "cat": r"\b(cat|cats|kitten|kittens)\b",
    "rabbit": r"\b(rabbit|rabbits|bunny|bunnies)\b",
}


@dataclass
class Step:
    """One entry in the agent's reasoning trace."""

    name: str
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Everything the agent produced for one request."""

    request: str
    status: str  # answered | scheduled | blocked
    answer: str
    intent: str = "question"
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    proposed_tasks: list[Task] = field(default_factory=list)
    plan_text: str = ""
    trace: list[Step] = field(default_factory=list)
    guardrail: str = "ok"
    revisions: int = 0

    def trace_text(self) -> str:
        """The trace as numbered, human-readable lines."""
        return "\n".join(
            f"{n}. [{s.name}] {s.detail}" for n, s in enumerate(self.trace, start=1)
        )


def parse_task_line(line: str) -> tuple[dict, dict]:
    """Split ``title | category | minutes | priority | frequency | rule`` into parts."""
    parts = [p.strip() for p in line.split("|")]
    fields = {
        "title": parts[0],
        "category": parts[1],
        "duration_minutes": int(parts[2]),
        "priority": parts[3],
        "frequency": parts[4],
    }
    rules = {}
    for rule in parts[5:]:
        key, _, value = rule.partition("=")
        rules[key.strip()] = value.strip()
    return fields, rules


def rule_allows(rules: dict, pet: Pet, request: str, has_med_task: bool) -> bool:
    """Check a TASK line's eligibility rules (min_age, max_age, needs) for a pet."""
    if "min_age" in rules and pet.age < int(rules["min_age"]):
        return False
    if "max_age" in rules and pet.age > int(rules["max_age"]):
        return False
    if rules.get("needs") == "medication":
        # Only when the owner mentions meds and no medication task exists yet.
        asked = bool(re.search(r"\b(med|meds|medication|medicine|pills?)\b", request))
        return asked and not has_med_task
    return True


def time_window(title: str) -> tuple[str, str]:
    """Preferred (earliest, latest) start window for a task title."""
    lowered = title.lower()
    for pattern, window in TIME_WINDOWS:
        if re.search(pattern, lowered):
            return window
    return ("07:00", "21:00")


class PawPalAgent:
    """Plans and answers pet-care requests over an owner's data.

    Args:
        retriever: Retrieval component; a default one over ``knowledge/`` is built.
        writer: Answer writer; chosen from ``PAWPAL_LLM`` when omitted.
        style: ``pawpal`` (specialized, cited) or ``baseline`` (plain).
        log_traces: Append every trace to ``logs/agent_traces.jsonl``.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        writer=None,
        style: str = "pawpal",
        log_traces: bool = True,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.writer = writer or get_writer()
        self.style = style
        self.log_traces = log_traces

    # ------------------------------------------------------------------ planning
    @staticmethod
    def plan(request: str, owner: Owner | None) -> tuple[str, list[Pet], list[str]]:
        """Decide intent, target pets and the tool sequence for a request."""
        lowered = request.lower()
        pets = list(owner.pets) if owner else []
        named = [p for p in pets if re.search(rf"\b{re.escape(p.name.lower())}\b", lowered)]
        by_species = [
            p for p in pets if re.search(SPECIES_WORDS.get(p.species, r"$^"), lowered)
        ]
        targets = named or by_species
        is_question = request.rstrip().endswith("?")
        if SCHEDULE_WORDS.search(lowered) and not is_question and pets:
            return "schedule", targets or pets, [
                "retrieve", "propose_tasks", "build_schedule", "critique", "answer"
            ]
        return "question", targets, ["retrieve", "answer"]

    # ------------------------------------------------------------------ tools
    def _retrieve(
        self, request: str, owner: Owner | None, targets: list[Pet], k: int
    ) -> list[Passage]:
        """Search once per target species (or once overall) and merge by best score."""
        names = [p.name for p in targets]
        species_list: list[str | None] = sorted({p.species for p in targets})
        if not species_list:
            lowered = request.lower()
            species_list = [s for s, pat in SPECIES_WORDS.items() if re.search(pat, lowered)]
        best: dict[str, Passage] = {}
        for species in species_list or [None]:
            for p in self.retriever.search(
                request, owner=owner, species=species, pet_names=names, top_k=k
            ):
                if p.source not in best or p.score > best[p.source].score:
                    best[p.source] = p
        return sorted(best.values(), key=lambda p: p.score, reverse=True)[:k]

    def _guides_for_pet(self, request: str, pet: Pet) -> list[Passage]:
        """Every task-bearing guide section for this pet's species, most relevant first.

        Sections come from the retriever's task index; each is scored against the
        request so the trace shows why it ranked where it did.
        """
        sections = self.retriever.task_sections(pet.species)
        scored = {
            p.source: p
            for p in self.retriever.search(request + " daily care", species=pet.species,
                                           top_k=len(self.retriever.guides))
        }
        ranked = [scored.get(p.source, p) for p in sections]
        return sorted(ranked, key=lambda p: p.score, reverse=True)

    @staticmethod
    def propose(request: str, pet: Pet, passages: list[Passage]) -> list[tuple[Task, str]]:
        """Build candidate tasks for one pet, with the source each came from.

        Skips tasks the pet already has, tasks its age rules exclude, and weekly
        tasks unless the request is about the week.
        """
        existing = {t.title.lower() for t in pet.tasks if not t.completed}
        has_med = any("medic" in t.title.lower() for t in pet.tasks)
        weekly_ok = bool(re.search(r"\bweek", request.lower()))
        proposals: list[tuple[Task, str]] = []
        seen: set[str] = set()
        for passage in passages:
            for line in passage.task_lines:
                fields, rules = parse_task_line(line)
                key = fields["title"].lower()
                if key in existing or key in seen:
                    continue
                if fields["frequency"] == "weekly" and not weekly_ok:
                    continue
                if not rule_allows(rules, pet, request.lower(), has_med):
                    continue
                seen.add(key)
                task = Task(**fields, due_date=date.today())
                task.pet_name = pet.name
                proposals.append((task, passage.source))
        return proposals

    @staticmethod
    def build(owner: Owner, proposals: list[Task]) -> Schedule:
        """Plan on a deep copy of the owner with the proposals added."""
        trial = copy.deepcopy(owner)
        by_name = {p.name: p for p in trial.pets}
        for task in proposals:
            by_name[task.pet_name].add_task(copy.copy(task))
        schedule = Schedule(day=date.today(), owner=trial)
        schedule.generate_plan()
        # Give untimed planned tasks a start time in their preferred window that
        # avoids busy blocks; fall back to the whole day if the window is full.
        for task in schedule.planned_tasks:
            if not task.time:
                earliest, latest = time_window(task.title)
                slot = (schedule.find_next_available_slot(task.duration_minutes,
                                                          earliest, latest)
                        or schedule.find_next_available_slot(task.duration_minutes,
                                                             "07:00", "21:00"))
                if slot:
                    task.time = slot
        return schedule

    # ------------------------------------------------------------------ main loop
    def run(self, request: str, owner: Owner | None = None) -> AgentResult:
        """Handle one request end to end and return the result with its trace."""
        started = time.perf_counter()
        trace: list[Step] = []

        screen = check_input(request)
        trace.append(Step("guardrail", f"input check -> {screen.category}",
                          {"ok": screen.ok, "matched": screen.matched}))
        if not screen.ok:
            result = AgentResult(
                request or "", "blocked", screen.message,
                confidence=1.0 if screen.category == "emergency" else 0.0,
                trace=trace, guardrail=screen.category,
            )
            return self._finish(result, started)
        dosage_mode = screen.category == "dosage"

        intent, targets, tools = self.plan(request, owner)
        trace.append(Step(
            "plan",
            f"intent={intent}; pets={[p.name for p in targets] or 'none'}; "
            f"tools={' -> '.join(tools)}",
            {"intent": intent, "pets": [p.name for p in targets], "tools": tools},
        ))

        passages = self._retrieve(request, owner, targets, k=4)
        trace.append(Step(
            "retrieve",
            f"{len(passages)} passages: "
            + (", ".join(f"{p.source} ({p.score})" for p in passages) or "none"),
            {"sources": [p.source for p in passages], "scores": [p.score for p in passages]},
        ))

        result = AgentResult(request, "answered", "", intent=intent, trace=trace,
                             guardrail=screen.category)
        next_step = None
        plan_issues: list[str] = []

        writer_query = request
        if intent == "schedule":
            next_step, plan_issues = self._schedule(request, owner, targets, passages,
                                                    result, trace)
            # The answer should explain the proposed tasks, so ask the writer about them.
            writer_query += " " + " ".join(t.title for t in result.proposed_tasks)
        else:
            next_step = self._question_next_step(passages, dosage_mode)

        answer = self.writer.write(writer_query, passages, self.style, dosage_mode,
                                   next_step)
        sources = [p.source for p in passages]
        check = check_answer(answer, sources, dosage_mode)
        trace.append(Step(
            "check_answer",
            ("answer passes checks" if check.ok else "; ".join(check.issues))
            + f" (writer={self.writer.name})",
            {"issues": check.issues},
        ))
        if not check.ok and sources:
            # One repair attempt: deterministic writer, doses stripped.
            answer = strip_doses(
                OfflineWriter().write(writer_query, passages, self.style, dosage_mode,
                                      next_step)
            )
            check = check_answer(answer, sources, dosage_mode)
            trace.append(Step("repair", "rewrote with offline writer -> "
                              + ("passes" if check.ok else "; ".join(check.issues))))

        result.answer = answer
        result.sources = sources
        result.confidence = self.confidence(passages, check.ok, plan_issues, dosage_mode)
        trace.append(Step("confidence", f"{result.confidence:.2f}"))
        return self._finish(result, started)

    @staticmethod
    def _question_next_step(passages: list[Passage], dosage_mode: bool) -> str:
        """Suggest a concrete follow-up: the top guide's routine task, if it has one."""
        if dosage_mode:
            return "call your vet's office and ask for the dose for your pet's weight."
        for p in passages:
            if p.kind == "guide" and p.task_lines:
                fields, _ = parse_task_line(p.task_lines[0])
                return (f"add '{fields['title']}' as a {fields['frequency']} task "
                        "(ask PawPal to plan today and it will schedule it).")
        return "ask your vet if this is still unclear."

    def _schedule(self, request, owner, targets, passages, result, trace):
        """Propose, build, critique and revise a schedule; returns (next_step, issues)."""
        candidates: list[tuple[Task, str]] = []
        for pet in targets:
            guide_hits = self._guides_for_pet(request, pet)
            pet_candidates = self.propose(request, pet, guide_hits)
            candidates += pet_candidates
            # Only sections that actually contributed a task become answer sources.
            used = {src for _, src in pet_candidates}
            known = {x.source for x in passages}
            passages.extend(p for p in guide_hits
                            if p.source in used and p.source not in known)
        trace.append(Step(
            "propose_tasks",
            f"{len(candidates)} candidate tasks: "
            + (", ".join(f"{t.title} for {t.pet_name} <- {src}" for t, src in candidates)
               or "none"),
            {"tasks": [t.title for t, _ in candidates]},
        ))

        proposals = [t for t, _ in candidates]
        schedule = self.build(owner, proposals)
        trace.append(Step(
            "build_schedule",
            f"planned {len(schedule.planned_tasks)} tasks, "
            f"{schedule.total_duration()}/{owner.available_minutes} min; "
            f"skipped {len(schedule.skipped_tasks)}",
        ))

        # Self-critique loop. Schedule.generate_plan is greedy by priority, so a
        # skipped high-priority task means high-priority care alone overflows the
        # budget. The only fix the agent may make on its own is to trade optional
        # proposals for time: drop its medium/low proposals and shorten ONE flexible
        # (exercise/enrichment) high-priority proposal to the freed slack. Anything
        # else (feeding, medicine, the owner's own tasks) is left to the owner.
        issues: list[str] = []
        adjustments: list[str] = []
        for attempt in range(MAX_REVISIONS + 1):
            skipped_high = [t for t in schedule.skipped_tasks if t.priority == "high"]
            critique = check_plan(schedule.total_duration(), owner.available_minutes,
                                  [f"{t.title} ({t.pet_name})" for t in skipped_high])
            issues = critique.issues
            trace.append(Step("critique",
                              "plan passes checks" if critique.ok else "; ".join(issues),
                              {"issues": issues}))
            if critique.ok or attempt == MAX_REVISIONS:
                break
            high_planned = sum(t.duration_minutes for t in schedule.planned_tasks
                               if t.priority == "high")
            slack = owner.available_minutes - high_planned
            skipped_keys = {(t.pet_name, t.title) for t in skipped_high}
            flexible = [t for t in proposals
                        if (t.pet_name, t.title) in skipped_keys
                        and t.category in FLEXIBLE_CATEGORIES]
            if not flexible or slack < MIN_FLEX_MINUTES:
                needed = high_planned + sum(t.duration_minutes for t in skipped_high)
                trace.append(Step(
                    "revise",
                    f"no safe automatic fix: high-priority care needs {needed} min, "
                    f"{owner.available_minutes} min available, slack {slack} min; "
                    "asking the owner to add time or move a task",
                ))
                break
            target = flexible[0]
            dropped = [t for t in proposals if t.priority != "high"]
            shortened = replace(target, duration_minutes=slack)
            proposals = [shortened if t is target else t
                         for t in proposals if t not in dropped]
            note = (f"shortened '{target.title}' for {target.pet_name} from "
                    f"{target.duration_minutes} to {slack} min")
            adjustments.append(note + " (below the guide's suggestion)")
            result.revisions += 1
            trace.append(Step(
                "revise",
                (("dropped optional " + ", ".join(f"{t.title} for {t.pet_name}"
                                                  for t in dropped) + "; ")
                 if dropped else "")
                + note + " and re-planned",
            ))
            schedule = self.build(owner, proposals)

        new_keys = {(t.pet_name, t.title) for t in proposals}
        result.proposed_tasks = [t for t in schedule.planned_tasks
                                 if (t.pet_name, t.title) in new_keys]
        result.plan_text = schedule.explain_plan()
        if adjustments:
            result.plan_text += "\nAdjusted by the agent: " + "; ".join(adjustments)
        if issues:
            result.plan_text += "\nNeeds your attention: " + "; ".join(issues)
        result.status = "scheduled"
        if result.proposed_tasks:
            first = min(result.proposed_tasks, key=lambda t: t.time or "99:99")
            next_step = (f"accept the plan to add {len(result.proposed_tasks)} new task(s), "
                         f"starting with '{first.title}' at {first.time or 'any time'}.")
        else:
            next_step = "your current tasks already cover today's care."
        return next_step, issues

    @staticmethod
    def confidence(passages: list[Passage], answer_ok: bool, plan_issues: list[str],
                   dosage_mode: bool) -> float:
        """Combine retrieval strength and check results into a 0-1 score."""
        guide_scores = [p.score for p in passages if p.kind == "guide"]
        if not guide_scores:
            return 0.1
        retrieval = min(1.0, max(guide_scores) / STRONG_SCORE)
        score = 0.2 + 0.6 * retrieval
        score += 0.2 if answer_ok else -0.3
        score -= 0.15 * len(plan_issues)
        if dosage_mode:
            score -= 0.1  # only general guidance is possible
        return round(max(0.0, min(1.0, score)), 2)

    def _finish(self, result: AgentResult, started: float) -> AgentResult:
        """Log a one-line summary and append the full trace to the JSONL log."""
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        log.info("request=%r status=%s intent=%s confidence=%.2f guardrail=%s ms=%s",
                 result.request[:80], result.status, result.intent, result.confidence,
                 result.guardrail, elapsed)
        if self.log_traces:
            try:
                LOG_DIR.mkdir(exist_ok=True)
                with open(LOG_DIR / "agent_traces.jsonl", "a") as fh:
                    fh.write(json.dumps({
                        "request": result.request,
                        "status": result.status,
                        "intent": result.intent,
                        "confidence": result.confidence,
                        "guardrail": result.guardrail,
                        "revisions": result.revisions,
                        "trace": [{"step": s.name, "detail": s.detail} for s in result.trace],
                        "answer": result.answer,
                    }) + "\n")
            except OSError as exc:  # logging must never break the answer
                log.warning("could not write trace: %s", exc)
        return result
