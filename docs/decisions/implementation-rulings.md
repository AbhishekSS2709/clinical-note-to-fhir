# SDD ledger — plan: docs/design/plans/2026-08-17-clinical-fhir-extraction-phases-0-4.md

Spec: docs/design/specs/2026-08-17-clinical-fhir-extraction-design.md (read, binding authority)
Branch: feat/fhir-extract-phases-0-4
Merge base: b17dd83

## Scope ruling

User goal (mid-turn): "complete all the phase/task".

Verified environment: Windows 11, Python 3.11.0, **no Java**, **no NVIDIA GPU**, vLLM has no
native Windows support. Tasks 3, 5, 6, 10, 11, 12, 13 contain steps that require Java, a GPU,
or an external corpus download.

Ruling: implement ALL 13 tasks' code and run every test that does not require GPU/Java/corpus.
Steps whose verification needs absent hardware are implemented and marked BLOCKED-ON-HARDWARE
in the ledger rather than skipped. — Why: maximises delivered work under the user's goal while
keeping claims honest; code is the deliverable, GPU runs are verification. — Cost if wrong:
GPU-path code is unexecuted, so runtime bugs there surface on first server run.

## Pre-flight conflict scan

### Cross-task interface pairs (shared file or interface)

| Producer | Consumer | Produces vs consumes | Finding |
|---|---|---|---|
| T2 profile | T3 synthea | `ClinicalRecord`, `Condition`, `Dosage`, `VitalObservation`, `VITAL_LOINC` | **E** — T3 reads `Condition.model_fields["clinical_status"].annotation.__args__` to validate a status; fragile introspection |
| T2 profile | T4 subset | `ClinicalRecord` | clean |
| T2 profile | T7 faithfulness | `ClinicalRecord` | clean |
| T2 profile | T9 metrics | `ClinicalRecord`, `validate_as_fhir` | clean |
| T2 profile | T10 baselines | `ClinicalRecord.model_json_schema()`, `VITAL_LOINC`, `VitalObservation` | clean |
| T3 synthea | T4 subset | `EncounterRecord` | clean |
| T3 synthea | T6 generate | `EncounterRecord`, `iter_bundles`, `parse_bundle` | clean |
| T4 subset | T6 generate | `select_subset(enc, rng)` | clean |
| T6 prompts | T6 generate | `build_prompt(enc, subset, rng)` | clean |
| T7 faithfulness | T8 dataset | `is_faithful` | clean |
| T7 faithfulness | T9 metrics | `_normalise`, `unanchored_facts` | **D** — T9 imports the private `_normalise` across module boundary |
| T8 dataset | T10 eval | `data/processed/*.jsonl` | clean |
| T9 metrics | T10 eval | `score`, `aggregate` | clean |
| T10 baselines | T11 train | `EXTRACT_INSTRUCTION` | clean |
| T10 baselines | T12 serve | `EXTRACT_INSTRUCTION` | clean |
| T11 train | T12 serve | adapter dir `outputs/adapters/qlora-8b` | clean |
| T12 serve | T13 web | `POST /extract` shape `{tuned, base, latency_ms}` | clean |

### Per-task internal consistency

| Task | Self-agreement (tests vs code vs files) | Finding |
|---|---|---|
| T1 scaffold | configs/Makefile/pyproject vs smoke test | clean |
| T2 profile | 7 tests vs models + `validate_as_fhir` | clean |
| T3 synthea | 4 tests need `tests/fixtures/bundle_sample.json`, produced by Synthea | **C** — fixture needs Java, unavailable |
| T4 subset | 6 tests vs policy | clean |
| T5 elmtex | no tests; deliverable is a recorded decision | needs corpus download — BLOCKED |
| T6 prompts/generate | 4 prompt tests pure-python; `generate.py` needs vLLM | **A** — shared `rng` consumed only for non-resumed encounters, so a resumed run produces different subsets/prompts than a fresh run; breaks the stated determinism constraint |
| T7 faithfulness | 6 tests vs filter | clean |
| T8 dataset | 4 tests vs splitter | clean |
| T9 metrics | 7 tests vs scorer | clean |
| T10 baselines/eval | 4 regex tests pure-python; LLM path needs vLLM | clean |
| T11 train | no unit tests; config + script | **B** — `RESPONSE_MARKER` defined and never used (dead code) |
| T12 serve | no unit tests; endpoint | clean |
| T13 web/README | no tests | clean |

### Rulings on scan findings

- **Ruling A (T6 determinism):** replace the single shared `random.Random(cfg["seed"])` with a
  per-encounter RNG seeded from `(seed, encounter_id)`, so a resumed run reproduces a fresh run
  exactly. — Why: Global Constraints require the dataset be regenerable from a seed, and the
  plan's own resume feature silently violated it. — Cost if wrong: none; strictly more
  deterministic than the plan text.
- **Ruling B (T11 dead code):** delete `RESPONSE_MARKER`. — Why: unused; the review rubric treats
  dead code as a defect. — Cost if wrong: none.
- **Ruling C (T3 fixture):** hand-author `tests/fixtures/bundle_sample.json` as a minimal but
  structurally faithful Synthea-shaped FHIR bundle (Patient + 2 Encounters + Condition +
  MedicationRequest + AllergyIntolerance + 2 Observations + Procedure) instead of generating it
  with Synthea. — Why: Java is absent; the fixture's purpose is exercising the parser, which a
  hand-authored bundle does equally well. — Cost if wrong: fixture may miss a real Synthea quirk,
  so the parser needs re-verification against genuine output on first server run. Flagged in T3's
  report as a required follow-up.
- **Ruling D (T9 private import):** promote `_normalise` to public `normalise` in
  `faithfulness.py`; keep no alias. — Why: cross-module import of a private name is a defect the
  reviewer would flag against plan-mandated code. — Cost if wrong: none.
- **Ruling E (T3 introspection):** export an explicit `CLINICAL_STATUSES: frozenset[str]` from
  `profile.py` and have T3 use it. — Why: `model_fields[...].annotation.__args__` breaks silently
  on any pydantic/typing change. — Cost if wrong: none.

## Ruling F — FHIR release namespace (found during setup, before Task 2 dispatch)

`fhir.resources` resolved to **8.3.0** on this machine. Its top-level namespace
(`fhir.resources.condition`, etc.) is **FHIR R5**, which REJECTS the R4 payloads the plan's
`_to_fhir_dicts` emits — verified: R5 MedicationStatement has no `medicationCodeableConcept`
and requires `medication`. Available release sub-namespaces: `R4B`, `STU3`.

Verified by direct execution: all five plan expansions (Condition, MedicationStatement,
AllergyIntolerance, Observation, Procedure) validate cleanly under `fhir.resources.R4B.*`.

**Ruling F:** Task 2's `validate_as_fhir` imports from `fhir.resources.R4B.*`, not the top-level
namespace; `pyproject.toml` pins `fhir.resources>=8.0`. — Why: the spec mandates FHIR R4, and
R4B is the R4-family release this package exposes; the plan's import paths would silently have
validated against R5 and failed every MedicationStatement. — Cost if wrong: R4B is FHIR 4.3, a
minor revision ahead of plain R4; if a downstream consumer demands exactly R4.0.1, the import
line changes but nothing else does.

## Environment findings (setup)

- Present: python 3.11.0, pydantic 2.12.4, rapidfuzz 3.14.3, pytest 9.0.2
- Installed during setup: fhir.resources 8.3.0 (+fhir-core 1.1.10), pyyaml, orjson, typer
- Absent, blocking: Java (Synthea), NVIDIA GPU + vLLM (generation/training/serving)

## Progress

Task 1: implemented (commit 3b275b1, base b17dd83) — 1/1 test passing, editable install worked.
  Review dispatched (rev-task1). Expected finding: `fhir.resources>=7.1.0` pin needs Ruling F bump to >=8.0.
Task 2: dispatched (impl-task2, base 3b275b1) with Rulings D, E, F carried in the brief.

Task 1: Ruling: the `fhir.resources>=7.1.0` pin (verified still present in pyproject.toml at 3b275b1)
  is a real finding against Ruling F, but Task 2's dispatch already instructs its implementer to
  change it to `>=8.0`. Do NOT open a Task 1 fix round for it; verify the corrected pin in Task 2's
  review instead. — Why: a one-line change already assigned to an in-flight task; a separate fix
  round would cost a full dispatch + re-review cycle for zero additional safety. — Cost if wrong:
  if Task 2 fails to make the change, the stale pin survives; Task 2's review is explicitly
  instructed to check it, and the repo would still work because 8.3.0 satisfies `>=7.1.0`.

Task 2: implemented (commit 524a97a, base 3b275b1) — 10/10 tests passing (controller-verified).
  Rulings applied and controller-verified: R4B imports present (profile.py:159-163),
  `CLINICAL_STATUSES` frozenset exported (profile.py:28), pin corrected to `fhir.resources>=8.0`.
  Ruling F regression test `test_medication_statement_uses_r4b_not_r5` present.
  Review dispatched (rev-task2).
Task 3: dispatched (impl-task3, base 524a97a) with Rulings C (hand-authored fixture, no Java),
  E (use CLINICAL_STATUSES not model_fields introspection), G (one vital per Observation).
Task 3: implemented (commit 0a2e1a4) — 15/15 passing. Ruling E verified at synthea.py:75.
  Fixture hand-authored: Patient 1, Encounter 2, Condition 2, MedicationRequest 2,
  AllergyIntolerance 1, Observation 4 (incl. 1 non-vital LOINC), Procedure 1, Immunization 1.
  CAVEAT CARRIED FORWARD: parser never validated against genuine Synthea output.
Task 4: implemented (commit 2545b9c). Task 7: implemented (commit 4ea5742). 27/27 passing.
  Ruling D verified: `def normalise` public at faithfulness.py:29, no `_normalise` alias.

Controller-verified directly (not taken on a reviewer's word):
  - `CLINICAL_STATUSES` == `typing.get_args(ClinicalStatus)` exactly (6 values). No drift.
  - All 8 `VITAL_LOINC` codes match the real standard LOINC codes.

Task 8+9: dispatched (impl-task8-9, base 4ea5742), batched — same shape, mutually independent.
  Ruling D carried (metrics must import public `normalise`). Task 8 Step 5 (`make data`)
  marked BLOCKED-ON-HARDWARE: requires data/interim/pairs.jsonl from Task 6 (GPU).

Task 8: implemented (commit fd5064e). Task 9: implemented (commit 304e70b). 38/38 passing.

Task 7: fix round 1/5 — background security review flagged fail-open paths in faithfulness.py.
  Controller assessment: mislabelled as security; it is a METRIC-CORRECTNESS defect and it is real.
  `_text_anchored` returns True for an empty/whitespace term. Task 9's `score()` reuses
  `unanchored_facts(note, pred)` as the hallucination detector on MODEL output, so a model
  emitting empty fields has those facts counted as anchored and therefore NOT hallucinated —
  the hallucination rate under-reports exactly when the model degenerates. Severity: Important.
  Routed to impl-task4-7 (original implementer, context intact) with 2 regression tests.
  Fix direction: fail CLOSED — an unverifiable fact counts as unanchored in both use sites.

Task 7: fix round 1/5 complete (commit c99e8a8) — fail-closed applied, 2 regression tests added,
  40/40 passing. Controller-verified at faithfulness.py:33-38.
Task 6: implemented (commit d7b1d91) — 45/45 passing. Ruling A verified: `_encounter_rng` with
  sha256(seed:encounter_id) at generate.py:31-40, used at line 60. Shared RNG removed.
  BLOCKED-ON-HARDWARE: brief Step 6 (20-encounter smoke generation) — needs vLLM + GPU.
Task 10: dispatched (impl-task10, base d7b1d91) with Rulings H (no faked LLM runs, no invented
  outputs/eval files) and J (keep the few-shot + constrained fair-fight baseline plumbed).

## Ruling I — Task 11 Step 7 cannot evaluate a bare adapter directory

Task 11 Step 7 runs `eval --system tuned --model outputs/adapters/qlora-8b`. That path is a PEFT
adapter directory, and vLLM cannot load a bare adapter as `model=` — it needs either the merged
weights or explicit LoRA support enabled. As written the step would fail at runtime on the server.

**Ruling I:** the tuned-model evaluation must run against the MERGED model produced by
`scripts/merge_adapter.py` (Task 12 Step 1), i.e. `--model outputs/merged/qlora-8b`. The merge
therefore precedes the tuned eval. Task 11's implementer must document this ordering in its report
and must not present Step 7's command as runnable-as-written. — Why: the plan's own task ordering
placed merge in Task 12 while Task 11 depended on its output. — Cost if wrong: none locally, since
both steps are BLOCKED-ON-HARDWARE; if wrong, the server run needs one extra merge invocation.

Task 10: implemented (commit c87830b) — 50/50 passing. Ruling H honoured (no outputs/ dir created).
Task 11: implemented (commit ec49560). Ruling B verified: 0 occurrences of RESPONSE_MARKER.
Task 12: implemented (commit 58f2029). Ruling I verified: Makefile `merge` target present,
  documented server order `data -> train -> merge -> eval -> serve` with the reason.
Task 13: implemented (commit 89d09b2). Ruling K verified: 7 `_pending_` placeholders,
  zero invented numbers in the results table.
Task 5:  implemented (commit 5dffd80). ELMTEX decision doc is a marked TEMPLATE, not a fake result.

ALL 13 TASKS IMPLEMENTED. 14 commits b17dd83..5dffd80, 41 files, 2112 insertions, 50 tests passing.
Final whole-branch review dispatched (final-review, opus).

## Final review — controller-found findings (fix wave dispatched: fix-final)

Reviewers proved unreliable this session (see note below), so the controller independently
audited the four highest-risk areas. Three real defects found, all in `metrics.py`:

1. **CRITICAL — fuzzy matching credits clinically dangerous confusions.** Verified by execution:
   `fuzz.token_sort_ratio('type 1 diabetes','type 2 diabetes') = 93.3`, above MATCH_THRESHOLD=88,
   so a model predicting "Type 1 diabetes" against gold "Type 2 diabetes" scores a TRUE POSITIVE.
   The headline F1 is inflated and rewards a serious clinical error. Fix: discriminator guard on
   differing digit-tokens and left/right laterality, applied before the ratio test.
2. **IMPORTANT — `_match` is first-match not best-match.** A mediocre candidate can consume a gold
   that a later prediction matches better, producing spurious FP+FN pairs; order-dependent.
3. **IMPORTANT — `macro_f1` penalises absent resource types.** `_f1(0,0,0)` returns 0.0 and is
   averaged in, so a PERFECT model on a split with no allergies/procedures scores macro_f1 = 0.6
   instead of 1.0. Fix: average only over types where tp+fp+fn > 0.

Verified sound (no defect found):
  - `split_by_patient`: each patient's rows are assigned to exactly one split via a single
    `extend` per patient. No patient can appear in two splits. Leakage invariant holds.
  - Lazy imports: no torch/transformers/peft/trl/vllm/datasets at module scope anywhere in
    `src/fhir_extract/`. (`scripts/merge_adapter.py` imports them at module scope, which is
    correct for a standalone script never imported by the package or tests.)

## Reviewer reliability note
rev-task1 and rev-task2 both went idle without delivering verdicts; chased via SendMessage.
Controller independently verified the load-bearing claims for both tasks (see above), so neither
task is being marked complete on an unverified assertion.

