# Critical fix wave — 2026-08-17

Seven defects found by a final whole-branch review of the clinical-note -> FHIR
extraction pipeline, fixed in this pass. Findings 1, 2, 3 were personally
verified by the reviewer by execution before this wave started; 4-7 were
verified by inspection/derivation and confirmed here by tests (except 4,
which cannot run without a GPU).

Test count: **54 -> 68** passing (`python -m pytest -q`), no existing test
weakened or deleted, no conflict between a fix and an existing test.

Commits:
- `0d4b91c` — faithfulness / dataset split / eval guard / metrics polarity / schema_validity (fixes 1, 2, 3, 5)
- `f30f115` — synthea BP component parsing / subset allergy fallback (fixes 6, 7)
- `6787e2c` — train.py startup crash, quantization, EOS, eval_strategy (fix 4)

---

## Fix 1 (critical) — `faithfulness.py` head-word rule admitted poisoned labels

**Problem:** `_text_anchored` anchored any label term whose first word was
>=5 characters, as long as that word appeared anywhere in the note. A label
`"Acute bronchitis"` was anchored by a note about `"acute viral
pharyngitis"`; `"Insulin glargine"` was anchored by `"insulin lispro"`. Since
this same function is the hallucination detector, a generated note that
dropped a fact but happened to share a generic head word with a *different*
condition passed the faithfulness filter and poisoned the training set.

**Fix:** Deleted the head-word branch in `src/fhir_extract/faithfulness.py`
entirely (no stop-list substitute). The existing `fuzz.partial_ratio >= 82`
fallback still covers genuine morphological variation (verified — no
existing test relied on the head-word branch; all had independent anchors
via exact-substring or the abbreviation table).

**Regression tests** (`tests/test_faithfulness.py`):
- `test_shared_generic_head_word_is_not_an_anchor`
- `test_shared_drug_stem_is_not_an_anchor`

Full suite re-run after the deletion: all existing faithfulness tests still
passed unmodified — no conflict.

---

## Fix 2 (critical) — `dataset.py` split fill-order starved val/test to zero

**Problem:** `split_by_patient` filled splits in dict order by absolute
count. With `configs/data.yaml` listing `train` first (target 8000) against
~8000 kept pairs, train consumed every patient and val/test_synthetic ended
up empty. `eval.py` defaults to `--split test_synthetic`, and
`metrics.aggregate([])` returned `{"n": 0, "micro_f1": 0.0, ...}` without
raising, so a run against an empty split silently produced a meaningless
0.0 results file instead of failing loudly.

**Fix:**
- `src/fhir_extract/dataset.py`: fill order is now `[k for k in ratios if k
  != "train"]`, with `train` as the unconditional overflow sink once every
  other split has hit its target. The per-patient wholesale assignment (no
  leakage across splits) is unchanged.
- `src/fhir_extract/eval.py`: raises `ValueError` naming the split file if
  the loaded split is empty, instead of proceeding to score against zero
  rows.
- `metrics.aggregate`: already returned `"n": len(rows)` (i.e. `0` for an
  empty list) without crashing — confirmed by inspection and by the new
  eval-side test; no change was needed there since the empty case was
  already reported honestly, just not treated as fatal by the caller.

**Regression tests:**
- `tests/test_dataset.py`: `test_held_out_splits_are_filled_before_train`,
  `test_no_patient_leaks_after_fill_order_change`
- `tests/test_eval.py`: `test_empty_split_raises_instead_of_writing_a_meaningless_result`

---

## Fix 3 (critical) — `metrics.py` polarity guard was incomplete

**Problem:** The existing digit/laterality discriminator did not catch
clinically opposed terms with high fuzzy-match ratios: `Hyperglycemia` /
`Hypoglycemia` (88.0, exactly at `MATCH_THRESHOLD`) and `Hypothyroidism` /
`Hyperthyroidism` (89.7) both scored as true positives.

**Fix:** Extended `_discriminators_conflict` in `src/fhir_extract/metrics.py`
with an opposed-prefix check for `hyper`/`hypo`, `acute`/`chronic`,
`primary`/`secondary`, detected as a **word-prefix** (`word.startswith(...)`)
rather than a substring search, so `hyperglycemia` vs `hypoglycemia` is
caught without false-triggering on unrelated substrings. `MATCH_THRESHOLD`
left at 88.

**Regression test:** `tests/test_metrics.py::test_polarity_discriminator_table`
(parametrized) — non-match asserted for hyperglycemia/hypoglycemia,
hypothyroidism/hyperthyroidism, hypercalcemia/hypocalcemia, and an
acute/chronic pancreatitis pair; genuine matches (case variants, exact
matches) still assert `True`.

---

## Fix 4 (critical) — `train.py` crashed at startup and never quantized

**Problem:** `SFTTrainer(..., model_init_kwargs={...})` passed a kwarg that
belongs to `SFTConfig`, not `SFTTrainer` — this raised `TypeError` before
step 0, and as a consequence `quantization_config` never reached the model
loader, so the run was never actually 4-bit even if the crash were papered
over.

**Fix (`src/fhir_extract/train.py`):**
- Load the model explicitly: `AutoModelForCausalLM.from_pretrained(cfg["model_id"],
  quantization_config=bnb, device_map="auto")`, then `SFTTrainer(model=model, ...)`.
  Dropped `model_init_kwargs`. `AutoModelForCausalLM` imported alongside the
  existing function-local transformers imports (no module-scope heavy imports
  added).
- `_format` now takes `tokenizer` and appends `tokenizer.eos_token` to every
  formatted example, so the model learns a stop signal (`packing=False`
  means TRL adds BOS but not EOS).
- `configs/train_qlora_8b.yaml`: added `eval_strategy: steps`, which
  `eval_steps: 200` requires to actually trigger evaluation.

**Verification:** No GPU available on this machine; per the reviewer's
instruction, torch/transformers/peft/trl/vllm were not installed and this
file was not executed. Verified by inspection and by AST-parsing the file
for syntax correctness. All 68 existing tests still pass — `train.py`'s
heavy imports remain function-local and untouched by the test suite.

---

## Fix 5 (critical) — `schema_validity` was trivially 1.0

**Problem:** `LLMBaseline.extract_batch` swallowed any JSON parse failure
into a bare `ClinicalRecord()`, and `validate_as_fhir(ClinicalRecord())`
iterates zero resources and returns `[]` (valid). Since `score()` only ever
saw an already-pydantic-valid record, nothing could ever fail schema
validity — a model emitting markdown-fenced JSON on every note scored
`schema_validity: 1.00`, identical to a perfect model.

**Fix:**
- `src/fhir_extract/baselines.py`: `extract_batch` now returns
  `list[tuple[ClinicalRecord, bool]]` — the bool is `True` on successful
  parse, `False` on the `ClinicalRecord()` fallback.
- `src/fhir_extract/eval.py`: threads the parse flag through; the `regex`
  system path wraps its output as `(record, True)` since it's not a JSON
  parse.
- `src/fhir_extract/metrics.py`: `score(..., parsed: bool = True)` — default
  `True` preserves all existing callers/tests — computes
  `"schema_valid": parsed and validate_as_fhir(pred) == []`.

**Regression test:** `tests/test_metrics.py::test_unparsed_prediction_is_never_schema_valid`
— `score(ClinicalRecord(), ClinicalRecord(), "note", parsed=False)["schema_valid"] is False`.

---

## Fix 6 (critical) — `synthea.py` dropped every blood pressure from real Synthea output

**Problem:** The Observation parser required both a top-level
`valueQuantity` and a code present in `VITAL_LOINC`. Real Synthea exports
blood pressure as a single Observation coded `85354-9` (not itself in
`VITAL_LOINC`) with **no top-level `valueQuantity`** — systolic (`8480-6`)
and diastolic (`8462-4`) live inside `component[]`. Both conditions failed,
so no BP ever reached a label. The hand-authored fixture used two separate
top-level Observations, which is why the existing tests passed while hiding
the real shape.

**Fix (`src/fhir_extract/synthea.py`):** Added a `component[]` branch ahead
of the existing top-level path: for an Observation with `component`, iterate
each component, map its `code.coding[].code` through `VITAL_LOINC`, and emit
one `VitalObservation` per matching component using that component's own
`valueQuantity`. The existing top-level path (for non-panel vitals) is
unchanged. Ruling G ("at most one vital per coding") is preserved at the
component level via the same `break` pattern used at the top level.

**New fixture:** `tests/fixtures/bundle_bp_component.json` — genuine Synthea
shape (Patient + 1 Encounter + 1 Observation coded `85354-9` with two
components).

**Regression test:** `tests/test_synthea.py::test_blood_pressure_panel_yields_both_components`
— asserts both `8480-6` and `8462-4` are extracted with the correct values.

---

## Fix 7 (important) — `subset.py` empty-label fallback skipped allergies

**Problem:** The "never emit an empty label" fallback tried a condition,
then vitals, then a medication — never an allergy. Since `parse_bundle`
assigns all allergies to the earliest encounter, an allergy-only encounter
combined with `ALLERGY_MENTION_PROB = 0.7` could roll to a fully empty
label (reviewer measured 54/200 seeds). An empty label passes the
faithfulness filter vacuously and teaches the model to return `{}` on notes
that do contain findings.

**Fix:** Added `elif src.allergies: sub.allergies = [src.allergies[0]]` as
the final link in the fallback chain in `src/fhir_extract/subset.py`.

**Regression test:** `tests/test_subset.py::test_allergy_only_record_is_never_empty`
— an allergy-only `ClinicalRecord` produces a non-empty subset across 100 seeds.

---

## Test counts

| Stage | Passing |
|---|---|
| Before this wave | 54 |
| After fixes 1, 2, 3, 5 (commit `0d4b91c`) | 66 |
| After fixes 6, 7 (commit `f30f115`) | 68 |
| After fix 4 (commit `6787e2c`, no test count change — train.py untestable without GPU) | 68 |

No fix conflicted with an existing test; none was weakened or deleted.
