# Model Card: PawPal AI

## Overview

| | |
|-|-|
| **System** | PawPal AI, a retrieval-grounded planning agent built on the PawPal+ scheduler (Module 2) |
| **Purpose** | Answer everyday pet-care questions and propose a daily care schedule for dogs, cats and rabbits, within an owner's available time |
| **Not for** | Diagnosis, emergencies, medication dosing, or species outside the knowledge base |
| **Components** | BM25 retriever over 7 guides (28 sections) and the owner's pet records; rule-based guardrails; a planning agent with a self-critique loop; an answer writer (offline extractive by default, optionally Claude `claude-opus-5-5` with few-shot prompting) |
| **Data** | `knowledge/*.md`: short care guides written in plain language from widely published general guidance (feeding frequency, exercise by age, toxic foods, grooming). They are not veterinary sources and have not been reviewed by a vet. |
| **Evaluation** | `eval/run_eval.py`: 15 labeled cases (15/15 pass), a retrieval ablation and a style comparison. `pytest`: 29 tests pass. |

## How it decides

1. Rules screen the request. Emergencies stop the flow, dose questions switch to
   "no numbers" mode, and prompt injection is refused.
2. The planner chooses between *question* and *schedule* and picks the target pets.
3. BM25 retrieves guide sections (species-filtered, with synonym expansion) and the
   owner's pet records.
4. For a schedule, tasks come only from human-written `TASK:` lines in the retrieved
   sections, filtered by age and medication rules. The original greedy scheduler places
   them, and a critique step checks that no high-priority care was dropped. If it was,
   the agent may shorten one exercise or enrichment activity, never below 10 minutes.
   Otherwise it flags the problem to the owner.
5. The writer phrases the answer only from the retrieved passages. An output check
   rejects answers with no source or with a dose, and repairs them.
6. Confidence = 0.2 + 0.6 × (best guide BM25 score / 6, capped at 1) + 0.2 if the
   answer passed its checks, minus 0.15 per unresolved plan issue and 0.1 in dosage mode.

## RAG enhancement: before and after

Retrieval was extended in three ways: **custom documents** (the guides in `knowledge/`,
chunked by `##` section and tagged with species), a **second source** (the owner's live
pet records, turned into text), and a **custom task index** (`Retriever.task_sections`),
which lets the planner see every routine for a species rather than only the sections that
match the request's words.

| Retriever (6 questions with a known correct section) | hit@4 | MRR |
|---|---|---|
| A. Keyword baseline: guides only, no synonyms, no species filter | 4/6 | 0.46 |
| B. + synonyms + species filter | 5/6 | 0.72 |
| C. + owner's pet records (multi-source) | 6/6 | 0.75 |

**Before and after examples:**

- *"When is Mochi's vet checkup?"* With A and B there is no correct answer anywhere,
  because the guides know nothing about Mochi. With C the top hit is
  `pet_record:Mochi`: "...Current pending tasks: Vet checkup (health, 60 min, medium
  priority, at 14:00)."
- *"What does Clover eat?"* With A, the word "eat" never matches the rabbit guide's
  "diet", so the Diet section ranks last of three rabbit sections. With B, "eat" expands
  to feeding/meal/diet and `rabbit_care.md#Diet` ranks first. The answer now says
  "About 80 percent of a rabbit's diet should be unlimited fresh grass hay...".
- *"Plan today for Biscuit"* (planner). Using only request-matched sections, the first
  version proposed no walk at all for a dog. With the task index, it proposes the
  age-appropriate *Gentle senior walk*.

## Specialization: baseline vs. PawPal style

The `pawpal` style is a constrained format, defined by two few-shot examples and rules
in `pawpal_ai/llm.py`: at most 3 bullets, each with a `[n]` citation, then a
`Next step:` line, then a fixed safety note. `baseline` is a plain answer from the same
passages. Measured over the 8 question cases (`python eval/run_eval.py`, section 3):

| Metric | Baseline | PawPal style |
|---|---|---|
| Answers with a citation `[n]` | 0% | 88% (the out-of-scope answer has nothing to cite) |
| Answers with a concrete "Next step" | 0% | 100% |
| Answers with the safety note | 0% | 100% |
| Answers with 1-3 bullets | 0% | 100% |
| Average words | 33 | 70 |

Same question, two styles:

```text
Q: What should I feed my cat Mochi, and how often?

Baseline:
Adult cats do well on two or more small measured meals a day. Kittens need three to four meals. Play before meals works well because it follows the natural hunt, eat, groom, sleep cycle.

PawPal style:
- Adult cats do well on two or more small measured meals a day [2].
- Kittens need three to four meals [2].
- Play before meals works well because it follows the natural hunt, eat, groom, sleep cycle [3].
Next step: add 'Feed breakfast' as a daily task (ask PawPal to plan today and it will schedule it).
Safety: this is general care guidance, not a diagnosis. Ask your vet about anything medical.
```

The specialized answers are about twice as long, because of the next step and safety
lines. That cost is accepted on purpose: the extra lines are what make an answer
checkable and safe. **Caveat:** these numbers come from the offline writer, which applies
the format deterministically. The Claude writer receives the same few-shot examples and
rules, but it was not run for this evaluation (no API key in the test environment), so
this card makes no claim about how closely Claude follows the format.

## Limitations and biases

- **Narrow, unreviewed knowledge.** There are three species and 28 short sections.
  Advice reflects common US pet-care guidance and leans toward indoor-companion norms.
  Breed differences are mostly ignored (a Husky and a Pug get the same "adult dog"
  routine), and exotic pets are not covered at all.
- **Keyword matching.** BM25 misses synonyms that are not in the small synonym table. A
  question phrased unusually can retrieve the wrong section, and the answer can still
  look confident.
- **Rule-based safety has gaps.** The emergency and dose patterns are regular
  expressions. "My dog got into the trash and is now really lethargic" is classified
  `ok`, not as an emergency. "Is half a pill okay for my cat?" does not trigger dosage
  mode, although the output check still strips any dose number from the answer.
- **The confidence score is a heuristic.** It measures how well the request words
  matched the guides, not whether the advice is right for this pet. Schedule requests
  score lower (around 0.6) even when the plan is good.
- **The scheduler is greedy.** Inherited from PawPal+, it treats a fixed appointment
  (Mochi's 60-minute vet visit) like an optional task and skips it when time is short.
  Recurring tasks are not filtered by due date.
- **The offline writer is extractive.** It sometimes picks a loosely related sentence;
  the senior-dog plan opens with a bullet about teeth.

## Could it be misused, and how is that prevented?

- **Using it instead of a vet.** This is the main risk. Mitigations: emergencies stop
  the flow and give a vet line plus the Poison Control number, every answer ends with a
  safety note, dosing is never answered with a number (input mode plus an output check
  plus a repair step), and out-of-scope questions say "I won't guess" with low
  confidence.
- **Prompt injection or off-topic use.** Injection phrasing is refused. The writer may
  only use the retrieved passages, and with Claude the system prompt restricts it to
  them.
- **Silent changes to the owner's data.** The agent works on a deep copy and only
  *proposes* tasks. Nothing is saved until the owner clicks "Add these tasks", and any
  change the agent makes to a guideline (a shortened walk) is spelled out.
- **What it does not prevent:** someone could still follow a general tip that is wrong
  for their specific pet. Better protection would need vet-reviewed content and
  per-pet medical context.

## What surprised me while testing reliability

- **My self-correction loop could never correct anything.** The first revise step
  dropped optional tasks to make room for a skipped high-priority task. Because the
  original scheduler is greedy by priority, high-priority tasks are always placed first, so dropping lower tasks never frees the space they need. The code looked reasonable. The trace showed the same critique failure after every revision, and that is how it was caught.
- **One word broke a correct answer.** "What does Clover eat?" failed the harness on its
  first run (14/15) because "eat" never reached "diet". The system then gave a
  confident-looking but empty reply.
- **The guardrails were the most reliable part.** All 5 guardrail cases passed on the
  first run. Every failure found was in retrieval or planning logic.
- **A first draft proposed "Give medication"** for a dog that already had "Joint
  medication". The medication rule had to check for any existing medication task, not
  only for an exact title match.

## Collaboration with AI during this project

I built the extension with Claude Code as a pair programmer. I set the direction
(extend PawPal+ with RAG, an agent and guardrails, keep it runnable without an API key,
and meet the rubric's stretch goals). Claude drafted modules, ran them, and read the
traces and evaluation output. Changes were judged by actual runs (`ai_main.py --trace`,
`eval/run_eval.py`, `pytest`) rather than by reading the code alone.

- **A suggestion I accepted:** run the agent on a `copy.deepcopy` of the owner and only
  *propose* tasks, with an explicit "Add these tasks" button in the Streamlit app. It
  costs one click, but it means the agent cannot corrupt saved data, and it put a real
  human-in-the-loop step into the architecture. There is a test for it
  (`test_agent_never_mutates_the_owner`).
- **A suggestion I rejected and revised:** the first self-critique design said "if a
  high-priority task is skipped, drop the agent's lowest-priority proposals and
  re-plan". In the household example, the trace showed the critique failing identically
  after every revision, because the greedy scheduler already places high-priority tasks
  first. It was replaced with a narrower rule: drop optional proposals *and* shorten one
  flexible exercise or enrichment task to the freed time (never below 10 minutes, never
  feeding or medicine). When even that cannot work, the plan says "Needs your attention"
  instead of pretending.

## Future improvements

- Have the guides reviewed by a veterinarian, add breed and size-specific sections, and
  add more species.
- Replace or augment BM25 with embeddings to close vocabulary gaps.
- Train a small classifier for emergencies on labeled examples, keeping the regular
  expressions as a floor, and add the "trash + lethargic" style of phrasing above.
- Teach the scheduler about fixed appointments and due dates, so the agent plans around
  them instead of skipping them.
- Calibrate confidence against human ratings of answer quality.
