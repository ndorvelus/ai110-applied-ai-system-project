"""Guardrails that run before and after the agent.

Input checks decide whether a request is safe to plan for at all:

* empty or oversized input is rejected,
* possible emergencies (poisoning, seizures, bloat, a blocked cat...) short-circuit the
  normal flow and tell the owner to contact a vet now,
* requests for medication doses are answered with general guidance only, never a
  number, because dosing is a veterinarian's decision,
* obvious prompt-injection phrasing is refused.

Output checks validate what the agent produced (budget respected, every answer backed
by a retrieved source, no dose numbers) so the self-critique step has something
concrete to act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 1000

EMERGENCY_PATTERNS = {
    "possible poisoning": r"\b(ate|eaten|eating|swallowed|got into|licked|chewed)\b.*\b("
    r"chocolate|grapes?|raisins?|xylitol|onions?|garlic|lil(y|ies)|ibuprofen|tylenol|"
    r"acetaminophen|advil|antifreeze|rat poison|pills?|gum)\b",
    "seizure": r"\bseiz(ure|ing|ures)\b|\bconvuls",
    "breathing trouble": r"\b(can'?t|cannot|struggling to|trouble|difficulty) breath",
    "collapse": r"\bcollaps(ed|e|ing)\b|\bunresponsive\b|\bpassed out\b",
    "bloat": r"\b(swollen|bloated|hard) (belly|stomach|abdomen)\b|\bretching\b",
    "urinary blockage": r"\bstraining\b.*\b(pee|urinat|litter)",
    "heavy bleeding": r"\bbleeding (heavily|a lot|won'?t stop)\b|\buncontrolled bleeding\b",
    "GI stasis": r"\brabbit\b.*\b(hasn'?t|not|stopped) (eaten|eating|pooped|pooping)\b",
}

DOSAGE_PATTERN = (
    r"\b(how (much|many)|what dose|dosage|dose of|mg|milligrams?)\b.*\b("
    r"benadryl|ibuprofen|tylenol|aspirin|medication|medicine|meds|pills?|mg|dose)\b"
)

INJECTION_PATTERN = (
    r"ignore (all|any|the|your|previous)[\w\s]*instructions|disregard (the|your) rules|"
    r"system prompt|you are now|jailbreak"
)

VET_LINE = (
    "Contact your veterinarian or an emergency animal hospital now. In the US you can "
    "also call the ASPCA Animal Poison Control Center at 888-426-4435."
)


@dataclass
class InputCheck:
    """Result of screening a user request."""

    ok: bool
    category: str = "ok"  # ok | empty | too_long | emergency | dosage | injection
    message: str = ""
    matched: list[str] = field(default_factory=list)


def check_input(text: str | None) -> InputCheck:
    """Screen a request before the agent plans anything."""
    if text is None or not text.strip():
        return InputCheck(False, "empty", "Please type a question or a planning request.")
    if len(text) > MAX_INPUT_CHARS:
        return InputCheck(
            False, "too_long",
            f"That request is {len(text)} characters; please keep it under "
            f"{MAX_INPUT_CHARS}.",
        )
    lowered = text.lower()
    if re.search(INJECTION_PATTERN, lowered):
        return InputCheck(
            False, "injection", "I can only help with pet-care questions and schedules."
        )
    hits = [name for name, pat in EMERGENCY_PATTERNS.items() if re.search(pat, lowered)]
    if hits:
        return InputCheck(
            False, "emergency",
            f"This sounds like it could be an emergency ({', '.join(hits)}). {VET_LINE} "
            "I won't build a routine schedule for this.",
            matched=hits,
        )
    if re.search(DOSAGE_PATTERN, lowered):
        # Not blocked: the agent still answers, but in "no doses" mode.
        return InputCheck(True, "dosage", "Dosing must come from your veterinarian.")
    return InputCheck(True)


@dataclass
class OutputCheck:
    """Problems found in an agent answer or plan; empty ``issues`` means it passed."""

    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def check_plan(planned_minutes: int, budget: int, skipped_high: list[str]) -> OutputCheck:
    """Validate a generated schedule against the owner's time budget."""
    result = OutputCheck()
    if planned_minutes > budget:
        result.issues.append(
            f"plan uses {planned_minutes} min but only {budget} min are available"
        )
    if skipped_high:
        result.issues.append("high-priority tasks were skipped: " + ", ".join(skipped_high))
    return result


DOSE_NUMBER = re.compile(r"\b\d+(\.\d+)?\s?(mg|milligrams?|ml|tablets?)\b")


def check_answer(answer: str, sources: list[str], dosage_mode: bool) -> OutputCheck:
    """Validate a free-text answer: grounded in sources and free of dose numbers."""
    result = OutputCheck()
    if not sources:
        result.issues.append("answer is not backed by any retrieved source")
    if dosage_mode and DOSE_NUMBER.search(answer.lower()):
        result.issues.append("answer contains a specific medication dose")
    return result
