"""Evaluation harness for PawPal AI.

Runs three experiments over the predefined inputs in ``eval/cases.json`` and prints a
summary (also written to ``eval/results.md``):

1. **Agent reliability** - every case runs through the full agent; each expectation
   (status, guardrail, required/forbidden tasks, cited source, answer text, confidence
   cap, revision count) is checked and reported as PASS/FAIL with the confidence score.
2. **Retrieval ablation** - for cases with an expected source, compares hit@4 and MRR
   for a plain keyword baseline, the tuned guide retriever (synonyms + species filter),
   and the full multi-source retriever (guides + the owner's pet records).
3. **Style specialization** - answers every question case in ``baseline`` and
   ``pawpal`` style and measures citation, next-step, safety-note and length metrics.

Usage:  python eval/run_eval.py          (exit code 1 if any case fails)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabulate import tabulate  # noqa: E402

from ai_main import demo_owner  # noqa: E402
from pawpal_ai import PawPalAgent, Retriever  # noqa: E402
from pawpal_ai.llm import SAFETY_NOTE  # noqa: E402

CASES = json.loads((ROOT / "eval" / "cases.json").read_text())


def check_case(case: dict, result) -> list[str]:
    """Return the list of failed expectations for one case (empty means pass)."""
    fails = []
    if result.status != case["expect_status"]:
        fails.append(f"status {result.status} != {case['expect_status']}")
    if "expect_guardrail" in case and result.guardrail != case["expect_guardrail"]:
        fails.append(f"guardrail {result.guardrail} != {case['expect_guardrail']}")
    titles = {t.title for t in result.proposed_tasks}
    for title in case.get("expect_tasks", []):
        if title not in titles:
            fails.append(f"missing task '{title}'")
    for title in case.get("forbid_tasks", []):
        if title in titles:
            fails.append(f"unexpected task '{title}'")
    if "expect_source" in case and case["expect_source"] not in result.sources:
        fails.append(f"source {case['expect_source']} not retrieved")
    if "expect_answer" in case and case["expect_answer"].lower() not in result.answer.lower():
        fails.append(f"answer lacks '{case['expect_answer']}'")
    if "forbid_answer_regex" in case and re.search(case["forbid_answer_regex"],
                                                   result.answer, re.IGNORECASE):
        fails.append("answer matches forbidden pattern")
    if "max_confidence" in case and result.confidence > case["max_confidence"]:
        fails.append(f"confidence {result.confidence} > {case['max_confidence']}")
    if result.revisions < case.get("expect_revisions_min", 0):
        fails.append(f"revisions {result.revisions} < {case['expect_revisions_min']}")
    if "expect_plan_text" in case and case["expect_plan_text"] not in result.plan_text:
        fails.append(f"plan text lacks '{case['expect_plan_text']}'")
    return fails


def run_agent_cases() -> tuple[list[list], int]:
    """Run every case through a fresh demo household; return table rows and passes."""
    agent = PawPalAgent(log_traces=False)
    rows, passed = [], 0
    for case in CASES:
        result = agent.run(case["input"], demo_owner())
        fails = check_case(case, result)
        passed += not fails
        rows.append([
            case["id"], "PASS" if not fails else "FAIL", result.status, result.guardrail,
            f"{result.confidence:.2f}", "; ".join(fails) or "-",
        ])
    return rows, passed


def rank_of(source: str, passages) -> int | None:
    """1-based rank of a source in a result list, or None if absent."""
    sources = [p.source for p in passages]
    return sources.index(source) + 1 if source in sources else None


def run_retrieval_ablation() -> list[list]:
    """Compare hit@4 and MRR across three retriever configurations."""
    owner = demo_owner()
    cases = [c for c in CASES if "expect_source" in c]
    plain = Retriever(use_pet_records=False, expand=False)
    tuned = PawPalAgent(retriever=Retriever(use_pet_records=False), log_traces=False)
    full = PawPalAgent(retriever=Retriever(), log_traces=False)
    configs = {
        "A. keyword baseline (guides only, no synonyms, no species filter)":
            lambda q: plain.search(q, top_k=4),
        "B. tuned guides (synonyms + species filter)":
            lambda q: tuned._retrieve(q, owner, tuned.plan(q, owner)[1], k=4),
        "C. multi-source (B + owner's pet records)":
            lambda q: full._retrieve(q, owner, full.plan(q, owner)[1], k=4),
    }
    rows = []
    for name, search in configs.items():
        ranks = [rank_of(c["expect_source"], search(c["input"])) for c in cases]
        hits = sum(r is not None for r in ranks)
        mrr = mean(1 / r if r else 0 for r in ranks)
        rows.append([name, f"{hits}/{len(cases)}", f"{mrr:.2f}"])
    return rows


def style_metrics(answers: list[str]) -> dict:
    """Format metrics that the specialized style is supposed to change."""
    def bullets(a: str) -> list[str]:
        return [ln for ln in a.splitlines() if ln.startswith("- ")]
    return {
        "cited (has [n])": mean(bool(re.search(r"\[\d+\]", a)) for a in answers),
        "has 'Next step:'": mean("Next step:" in a for a in answers),
        "has safety note": mean(SAFETY_NOTE in a for a in answers),
        "<= 3 bullets": mean(0 < len(bullets(a)) <= 3 for a in answers),
        "avg words": mean(len(a.split()) for a in answers),
    }


def run_style_comparison() -> tuple[list[list], dict]:
    """Answer the question cases in both styles and tabulate the metrics."""
    questions = [c["input"] for c in CASES if c["expect_status"] == "answered"]
    out, examples = {}, {}
    for style in ("baseline", "pawpal"):
        agent = PawPalAgent(style=style, log_traces=False)
        answers = [agent.run(q, demo_owner()).answer for q in questions]
        out[style] = style_metrics(answers)
        examples[style] = answers[0]
    rows = []
    for metric in out["baseline"]:
        b, p = out["baseline"][metric], out["pawpal"][metric]
        fmt = (lambda v: f"{v:.0f}") if metric == "avg words" else (lambda v: f"{v:.0%}")
        rows.append([metric, fmt(b), fmt(p)])
    return rows, {"question": questions[0], **examples}


def main() -> int:
    agent_rows, passed = run_agent_cases()
    retrieval_rows = run_retrieval_ablation()
    style_rows, example = run_style_comparison()
    confidences = [float(r[4]) for r in agent_rows if r[2] != "blocked"]

    sections = [
        ("1. Agent reliability", tabulate(
            agent_rows, ["case", "result", "status", "guardrail", "conf", "failed checks"],
            tablefmt="github")),
        ("2. Retrieval ablation (expected source in top 4)", tabulate(
            retrieval_rows, ["retriever", "hit@4", "MRR"], tablefmt="github")),
        ("3. Style specialization (question cases)", tabulate(
            style_rows, ["metric", "baseline", "pawpal (few-shot)"], tablefmt="github")),
    ]
    summary = (
        f"{passed}/{len(agent_rows)} cases passed; average confidence on non-blocked "
        f"cases {mean(confidences):.2f}; "
        f"{sum(r[2] == 'blocked' for r in agent_rows)} requests blocked by guardrails."
    )
    report = ["# PawPal AI evaluation results", "", f"**Summary:** {summary}", ""]
    for title, table in sections:
        report += [f"## {title}", "", table, ""]
    report += [
        "## Example: same question, two styles", "",
        f"Question: {example['question']}", "",
        "Baseline:", "```text", example["baseline"], "```", "",
        "PawPal style:", "```text", example["pawpal"], "```", "",
    ]
    (ROOT / "eval" / "results.md").write_text("\n".join(report))

    for title, table in sections:
        print(f"\n== {title} ==\n{table}")
    print(f"\nSUMMARY: {summary}")
    print("Full report written to eval/results.md")
    return 0 if passed == len(agent_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
