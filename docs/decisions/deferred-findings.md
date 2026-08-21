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

## Resolved

- **Completion-only loss masking is now implemented and self-verifying.**
  `train.py` builds conversational (`messages`) examples, patches the tokenizer's chat
  template with `{% generation %}` / `{% endgeneration %}` markers via
  `ensure_generation_markers` when they're missing (Qwen3's stock template lacks them),
  and enables TRL's `assistant_only_loss` / `completion_only_loss` (or, on the oldest TRL
  API, `DataCollatorForCompletionOnlyLM`) via `_configure_completion_only_loss`. A hard
  runtime assertion (`_verify_masking`, gated by the `verify_masking: true` config flag)
  decodes one training batch's labels and raises before training starts if the instruction
  text is visible in the unmasked portion. **This assertion has still never executed on a
  GPU** — it was written from the documented TRL API shape (no GPU/torch/trl available in
  this environment) and remains unconfirmed against a real model/tokenizer until Task 11
  Step 4 runs on the server; see `docs/SERVER-RUN-CHECKLIST.md`.

## Known-unconfirmed

- **The Synthea parser has still only been validated against hand-authored fixtures.**
  The BP `component[]` fix (C6) was written from the documented real shape, not from
  genuine output. Step 1 of `docs/SERVER-RUN-CHECKLIST.md` remains mandatory.
- **The manifest records seed, model, drop rate and distinct-n, but not the generation
  sampling params or the subset-policy constants.** Changing `KEEP_CONDITIONS` silently
  changes the corpus while the recorded seed stays the same.

## Unverified API assumptions in `train.py` (check these FIRST on the server)

The Qwen3 training patches were written offline with no torch/trl/transformers installed.
Every item below is an assumption, not a verified fact. Check them before a long training run.

1. TRL exposes `SFTConfig(assistant_only_loss=True)`, or an older one `completion_only_loss=True`,
   as boolean kwargs with those exact names.
2. `from trl import DataCollatorForCompletionOnlyLM` still exists on the oldest supported TRL and
   takes `response_template=` + `tokenizer=`.
3. `return_assistant_tokens_mask` masking genuinely keys off `{% generation %}` /
   `{% endgeneration %}` Jinja tags (documented HF behaviour, unverified against the installed version).
4. **HIGHEST RISK — Qwen3's real chat template.** `ensure_generation_markers` assumes the assistant
   branch is a flat `{% if/elif role == 'assistant' %} ... {% endif %}` with no confusing nesting.
   The test fixture is a plausible ChatML approximation, NOT Qwen3's actual template — it could not
   be fetched offline. Qwen3 is a hybrid thinking model and its template very likely contains
   nested `<think>` / tool-call blocks inside that branch, which can make the depth-tracking scanner
   wrap the wrong span or raise "could not find closing tag". **Print the patched template and eyeball
   it before the first real run.**
5. `AutoModelForCausalLM.from_pretrained` exposes `dtype`/`torch_dtype` to `inspect.signature`.
   If a decorator erases the signature, the shim guesses `"dtype"` via a `**kwargs` heuristic.
6. That passing BOTH `warmup_ratio` and `warmup_steps` is the actual v5 failure mode (implemented
   defensively; the original diff was unavailable).
7. `trainer.get_train_dataloader()` and batch `["labels"]` are still the right accessors.

Mitigation: `_verify_masking()` runs automatically before training and raises if the instruction
text leaks into the unmasked span — so assumptions 1-4 fail LOUDLY at step 0 rather than silently
producing a badly-trained model. That assertion has never executed on a GPU.


---

# Status update — 2026-08-21 (after the full GPU run)

## Closed by execution

| # | Finding | How it was closed |
|---|---|---|
| I6 | `make data` had no `synthea`/`generate` targets, so the documented reproduce path failed | Makefile rewritten with the full order: `synthea -> generate -> data -> train -> serve-ab -> eval`, plus `elmtex` and `benchmark` |
| I7 | `outputs/serving_benchmark.json` cited but never produced | `scripts/benchmark_serving.py` added; measured 326 prompt tokens/note vs 8,100 for few-shot, $83.64 vs $191.57 per 1M notes |
| I9 | Unbounded pins across known API renames | `trl` 1.10 renamed `SFTConfig.max_seq_length` to `max_length` and it DID fire. `resolve_max_length_kwarg` emits whichever the installed version accepts, and `unsupported_kwargs` pre-flights the whole training block against the signature before the model loads |
| M5 | `report_to="wandb"` unconditional | Config-driven, default `none`. wandb was not installed on the server, so this raised at startup rather than hanging — either way it blocked the first launch |

**The completion-only masking assertion has now executed on a GPU.** It was the
highest-risk unverified assumption (#4: Qwen3's real chat template vs our
marker-patching, tested only against a hand-written approximation). It passed on
both corpora:

```
[verify_masking] 693/809 label tokens masked   (synthetic)
[verify_masking] 1015/1776 label tokens masked (ELMTEX)
```

That output also revealed a bug it was not looking for: the supervised span
begins `<think>

</think>`, because Qwen3's template emits the think tags
inside the assistant turn. `parse_record` handled neither, so every fine-tuned
prediction would have failed to parse and scored 0.0 — a working model reported
as a failed experiment. Fixed with a reasoning-block stripper.

**The Synthea parser has now been run against genuine output**: 180,093 real
encounters parsed, 28,600 pairs generated. The C6 `component[]` fix for blood
pressure held up on real bundles.

## Still open

| # | Finding | Status |
|---|---|---|
| I4 | `_number_anchored` matches the literal numeral, so the unit-conversion generation axis is systematically filtered out | Open. Fails closed (drops pairs), so it inflates the drop rate rather than poisoning labels |
| I8 | `synthea.py` hard-codes `dosage=Dosage()`, so every medication label has null dose/unit/route | Open — and now known to matter: it is exactly why ELMTEX medication scoring is incomparable, since ELMTEX concatenates dose into the name |
| M1 | "field-level F1" is really *name*-level F1 | Open. The README says name-level for ELMTEX but the synthetic tables should say so too |
| M2 | Vitals matching key omits the unit | Open |
| M3 | `hallucination_rate` / `omission_rate` divide facts by rows, so they can exceed 1.0 | Open. The README reports them as "facts/note", which is what they are, but the key names still say `_rate` |
| M4 | `serve.py` resolves paths against CWD at import | Open |
| M6 | `baselines.py` temperature regex matches a bare `T`; units hard-coded | Open. Affects the regex baseline only |

## New, found during the run

- **Result filenames collided.** Every eval wrote `{system}_{shots}shot_{split}`,
  so the fine-tuned run silently overwrote the base baseline mid-run. `result_tag`
  now includes the model and the constrained/schema-hint flags. Any axis that
  changes the number changes the filename.
- **The zero-shot baseline was a strawman.** `EXTRACT_INSTRUCTION` names the five
  keys but never describes the field structure, so the base model invented its own
  shape and scored 0.000. Baselines now receive the schema rendered from the
  Pydantic model.
- **The fine-tuned model degenerates into unbounded repetition on real reports.**
  Not a truncation artifact — 4096 tokens changes nothing. See
  `elmtex-evaluation.md`.
