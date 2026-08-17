# Deferred findings — open items from the final whole-branch review

Date: 2026-08-17. All 6 Critical findings were fixed (see `critical-fixes-2026-08-17.md`)
and verified by execution. The items below were **triaged as non-blocking and left open**.
None is a silent discard — each is recorded here with why it was deferred.

Fix these before publishing any number, not before running the pipeline.

## Deferred — affect metric meaning (fix before the README table is populated)

| # | Finding | Why deferred | Cost if left |
|---|---|---|---|
| I4 | `_number_anchored` matches the literal numeral only, so the "inconsistent units" generation axis (lbs/kg, F/C — ~20% of the corpus) is systematically filtered out. Spec §4.5 requires unit-conversion awareness. | Fails closed (drops pairs) rather than poisoning labels. | Inflates and misattributes the drop rate; deletes a diversity axis added specifically to stop the student overfitting to one voice. |
| I8 | `synthea.py` hard-codes `dosage=Dosage()`, so every medication label has null dose/unit/route/frequency. `manifestation` reads `.get("text","")` where Synthea supplies `coding[].display`, yielding empty strings. | Metrics never score these fields, so the gap is invisible in every current number. | Spec §2 lists dosage in the profile. Either populate it from `dosageInstruction[0]` or delete it from the model — do not ship an always-null field. |
| M1 | "field-level F1" is really *name*-level F1: `clinical_status`, `onset_date`, medication `status`, `performed_date`, `criticality`, `manifestation` are never scored. A model marking every condition `resolved` scores identically to one that gets status right. | Renaming/expanding changes every baseline number; must be decided before baselines run. | The README column header overstates what is measured. |
| M2 | Vitals matching key omits the unit, so 37.0 °C is credited against 37.0 °F. | Same as above. | Spec §7 says units match exactly. |
| M3 | `hallucination_rate` / `omission_rate` divide a fact count by a row count, so they can exceed 1.0 while the README presents them as percentages. | Cosmetic until published. | Rename to `*_per_note`, or divide by fact totals. |

## Deferred — runtime landmines on hardware-blocked paths (fix before the server run)

| # | Finding | Cost if left |
|---|---|---|
| I6 | `make data` reads `data/interim/pairs.jsonl`, but no Make target produces `data/raw/synthea` or runs `generate`. The README's six-command reproduce path fails with `FileNotFoundError` and no explanation. | Add `synthea:` and `generate:` targets at the head of the documented order. |
| I7 | `outputs/serving_benchmark.json` is cited by the README but no script emits it. | Add `scripts/benchmark_serving.py` (tok/s at batch 1/8/32, p50/p95, peak VRAM, cost per 1M notes). |
| I9 | Unbounded pins across known API renames: `vllm>=0.6` (`guided_decoding`/`GuidedDecodingParams` → `structured_outputs`/`StructuredOutputsParams` in V1) and `trl>=0.10` (`SFTConfig.max_seq_length` → `max_length`, which `train.py`'s `**t` would hit as a `TypeError`). Separately, `serve.py` builds two vLLM engines in one process, which V1 does not reliably support. | Upper-bound both pins; run the two models as separate processes. |
| M4 | `serve.py` resolves `configs/serve.yaml` and `StaticFiles("web")` against CWD at import, so importing outside the repo root raises. | Resolve paths relative to the module. |
| M5 | `train.py` sets `report_to="wandb"` unconditionally — a headless run hangs at step 0 without `WANDB_API_KEY`. | Make it configurable. |
| M6 | `baselines.py` temperature regex matches a bare `T` before digits in unrelated text; units are hard-coded (F/kg/cm) regardless of what the note says. | Tighten the pattern; read units from the match. |

## Known-unconfirmed

- **Completion-only loss masking is NOT wired up.** `train.py`'s `_format` trains on the
  instruction text as well as the answer. The plan calls this the highest-value detail and
  Task 11 Step 4 exists to verify it — but that step is hardware-blocked, so this is
  *unconfirmed rather than merely unrun*. Resolve it before spending GPU hours.
- **The Synthea parser has still only been validated against hand-authored fixtures.**
  The BP `component[]` fix (C6) was written from the documented real shape, not from
  genuine output. Step 1 of `docs/SERVER-RUN-CHECKLIST.md` remains mandatory.
- **The manifest records seed, model, drop rate and distinct-n, but not the generation
  sampling params or the subset-policy constants.** Changing `KEEP_CONDITIONS` silently
  changes the corpus while the recorded seed stays the same.
