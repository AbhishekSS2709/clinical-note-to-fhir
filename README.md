# Clinical Note → FHIR

[![CI](https://github.com/AbhishekSS2709/clinical-note-to-fhir/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhishekSS2709/clinical-note-to-fhir/actions/workflows/ci.yml)

Extracts structured FHIR R4 clinical records (conditions, medications, allergies, vitals, procedures) from free-text clinical notes, via a LoRA fine-tuned Qwen3-8B served side-by-side with its untouched base model.

Training data is manufactured by **reverse generation**: Synthea emits valid FHIR bundles, an LLM writes clinical notes from them, and the bundle subset *is* the label — correct by construction, with no human annotation and no teacher model to inherit errors from.

**The interesting result is the failure and the fix.** Trained on synthetic notes, the model reaches **micro-F1 0.996** on its synthetic test split and **0.208** on real clinical reports — *below* the un-finetuned base model's 0.445, degenerating into unbounded repetition. Retrained on real reports (v2), the same recipe reaches **0.706**, beating the base model by 59% relative. [Where it fails](#where-it-fails-and-why) has the diagnosis; [v2](#v2-the-fix) has the fix.

## Results

Every number below is rendered from `outputs/eval/*.json` by `python scripts/results_table.py` — none is typed by hand, and the script regenerates these tables (plus per-resource breakdowns) so any drift between the README and the result files is visible. All systems share one vLLM endpoint, so sampling, parser and server settings are identical across rows.

### In-distribution — synthetic test split (n=517)

| System | Micro-F1 | Macro-F1 | Schema validity | Hallucinated facts/note |
|---|---:|---:|---:|---:|
| Regex baseline | 0.409 | 0.097 | 1.000 | 0.000 |
| Qwen3-8B 0-shot (schema in prompt) | 0.257 | 0.390 | 0.884 | 0.151 |
| Qwen3-8B 0-shot + constrained decoding | 0.259 | 0.391 | 0.888 | 0.151 |
| Qwen3-8B 5-shot | 0.898 | 0.569 | 0.983 | 0.269 |
| **Qwen3-8B + LoRA bf16 (this project)** | **0.996** | **0.786** | **1.000** | **0.002** |
| Qwen3-8B + QLoRA 4-bit (ablation) | 0.992 | 0.775 | 0.994 | 0.000 |

### Allergy slice (n=400)

`AllergyIntolerance` appears in exactly one row of the main test split, so a dedicated enriched slice was built to make it measurable at all.

| System | Micro-F1 | Macro-F1 | **Allergy F1** |
|---|---:|---:|---:|
| Qwen3-8B 0-shot | 0.280 | 0.501 | 0.753 |
| Qwen3-8B 5-shot | 0.865 | 0.697 | 0.748 |
| **Qwen3-8B + LoRA bf16** | **0.993** | **0.980** | **0.963** |

Trained on ~100 allergy examples out of 17,696, the fine-tune still beats few-shot by 0.22 F1 on that type.

### Out-of-distribution — ELMTEX real clinical reports (n=599)

Scored on `conditions,procedures` only. See [docs/decisions/elmtex-evaluation.md](docs/decisions/elmtex-evaluation.md) for why medications are excluded and why the unannotated types must be excluded rather than scored as empty.

| System | Micro-F1 | Macro-F1 | Schema validity | Omitted facts/note |
|---|---:|---:|---:|---:|
| Qwen3-8B 0-shot (base) | 0.445 | 0.434 | 0.649 | 5.56 |
| Qwen3-8B 5-shot | 0.379 | 0.382 | 0.848 | 6.76 |
| Qwen3-8B + LoRA bf16 | 0.208 | 0.173 | 0.639 | 8.45 |
| Qwen3-8B + QLoRA 4-bit | 0.199 | 0.178 | 0.591 | 8.53 |
| **Qwen3-8B + LoRA on real reports (v2)** | **0.706** | **0.684** | **1.000** | **2.88** |

## Where it fails, and why

The fine-tuned model does not merely score lower on real text — it **degenerates into unbounded repetition**. Measured on 16 ELMTEX reports with greedy decoding and identical prompts:

| Model | max_tokens | Hit token cap | Parsed | Median output tokens |
|---|---:|---:|---:|---:|
| base | 2048 | 0 | 16/16 | 623 |
| base | 4096 | 0 | 15/16 | 623 |
| fine-tuned | 2048 | **10** | 5/16 | 2048 |
| fine-tuned | 4096 | **10** | 5/16 | 4096 |
| **v2 (trained on real reports)** | 2048 | **0** | **16/16** | **412** |

The base model terminates at a median of 623 tokens. The fine-tuned model consumes *whatever budget it is given*, repeating entries until it runs out. Doubling the budget changes nothing, and constrained decoding does not help either — the loop happens inside the JSON arrays, which the schema permits.

The raw output shows what it actually learned:

```json
{"medication_text": "100 MG/ML Hydrocortisone Injectable Powder/Injection",
 "dosage": {"dose": null, "unit": null, "route": null, "frequency": null},
 "status": "active"}
```

Synthea's verbose RxNorm display names, SNOMED `(finding)` tags, every optional field spelled out as `null`, and an `onset_date` it invents. It learned the *generator's surface form*, not the extraction task. Synthetic labels carry a median of 6 facts; real reports carry ~12, and the model never learned to terminate a long list.

**This is the argument for external validation.** In-distribution, every metric said the project was a success.

## What the demo shows

`make serve` renders both models side by side. One hand-written note reproduces the entire finding:

> 58-year-old man with type 2 diabetes mellitus and hypertension. BP 148/92 mmHg, HR 88. Continues metformin 500 mg twice daily. Allergic to penicillin. ECG performed today.

| | fine-tuned | base |
|---|---|---|
| conditions | `[]` — **missed both** | `Type 2 diabetes mellitus`, `Hypertension` |
| medications | `Metformin 500 MG Oral Tablet` | `metformin` |
| allergies | `Penicillin (substance)` | `penicillin` |
| vitals | `8480-6=148` systolic, `8462-4=92` diastolic, `8867-4=88` | `8302-2=148` **body height**, `8310-5=148` **temperature**, `8462-4=92`, `8867-4=88` |
| procedures | `ECG (procedure)` | `ECG` |

The fine-tune assigns correct LOINC codes; the base model codes a blood pressure of 148 as body height *and* body temperature. That is the value fine-tuning adds, and no prompt engineering produced it.

On the same note the fine-tune drops both diagnoses. The note is ordinary prose rather than Synthea-shaped text, so it is off-distribution — and conditions collapse exactly as the ELMTEX numbers predict.

Both panes are prompted fairly: the tuned model gets the prompt it was trained on, the base model additionally gets the schema, since it has never seen the field structure.

## The LoRA vs QLoRA ablation

Both runs are identical except precision — same effective batch (32), learning rate (2e-4), rank (16), target modules, epoch count and data.

| | bf16 LoRA | QLoRA 4-bit |
|---|---:|---:|
| eval loss @200 / @400 / @553 | 0.0056 / 0.0049 / **0.0048** | 0.0057 / 0.0050 / 0.0050 |
| synthetic micro-F1 | **0.996** | 0.992 |
| allergy-slice micro-F1 | **0.993** | 0.989 |
| ELMTEX micro-F1 | 0.208 | 0.199 |

bf16 wins consistently but by ~0.004 F1 — negligible. More usefully: **precision choice does not affect the generalization failure at all.** Both overfit the generator identically.

*No wall-clock comparison is reported.* The two runs sat on differently-loaded GPUs of a shared cluster (31 s/step vs 25–34 s/step), so those numbers would measure the cluster, not the method.

## Why fine-tuning, not RAG

This task needs a **fixed output schema** and **cheap, high-volume inference** — retrieval solves neither. RAG earns its keep when an answer depends on fresh or per-query external knowledge; extracting the same five FHIR resource types from a self-contained note is a closed, schema-constrained transformation, not a knowledge-lookup problem. Fine-tuning bakes the schema into the weights: the fine-tuned model spends **326 prompt tokens per note** where 5-shot spends **8,100**, because it carries neither exemplars nor a schema block. That is the whole serving argument, and it is measured below.

### Serving cost (one A6000, vLLM, `outputs/serving_benchmark.json`)

| System | Notes/sec @32 | p95 latency | Prompt tokens/note | $/1M notes |
|---|---:|---:|---:|---:|
| **Fine-tuned 0-shot** | **2.66** | **14.4 s** | **326** | **$83.64** |
| Base 0-shot + schema | 2.03 | 21.2 s | 1,078 | $109.25 |
| Base 5-shot | 1.16 | 39.4 s | 8,100 | $191.57 |

2.3x the throughput of few-shot at 2.3x lower cost with a 2.7x faster tail. Cost assumes $0.80/GPU-hour and scales linearly — substitute your own rate.

This advantage is real but applies **in-distribution only**; on real reports the fine-tuned model is both slower to terminate and less accurate.

## How the data was built

Real clinical notes come with no ground truth. Rather than pay for labeling, or trust a bigger model's guesses as if they were gold, this pipeline runs the extraction task **backwards**:

```
Synthea  ──generates──▶  valid FHIR bundle  ──(this IS the gold label)
                               │
                               ▼
                      subset selection
                               │
                               ▼
                 Qwen3-8B writes a clinical note
                      conditioned on that subset
                               │
                               ▼
                    (note, label) training pair
```

Synthea (MITRE, Apache 2.0) generates synthetic patients with valid FHIR R4 bundles — the label is correct by construction, with no teacher model and no circular evaluation. This is the *asymmetry principle* from **SynthIE** (Josifoski et al., [arXiv:2303.04132](https://arxiv.org/abs/2303.04132)): for tasks with structured outputs, generating plausible text *from* a target structure is far easier for an LLM than extracting the structure from text, so the easy direction is run to manufacture data for the hard one.

**Subset selection is the critical correctness detail.** A Synthea bundle is a patient's *entire* history; a real clinical note only ever mentions a slice of it. Labeling a generated note with the full bundle would train the model to emit facts absent from the text — i.e. train it to hallucinate. Instead, a realistic subset is sampled first, *that subset becomes the label*, and the note is generated conditioned on exactly that subset.

**Both directions of faithfulness are then filtered.** Of 28,600 generated pairs, 19,123 survived:

| | rate | what it catches |
|---|---:|---|
| `drop_rate_missing` | 25.9% | note omits a labelled fact |
| `drop_rate_invented` | 8.5% | note asserts a fact absent from the label |

The reverse check matters more than it looks: those 8.5% contain drugs and findings the label never mentioned, and every one would have taught the model that inventing plausible clinical detail is rewarded.

Splits are **patient-level**, verified as disjoint by execution rather than assumed.

## Honest limitations

- **This is a narrowed FHIR profile, not full FHIR conformance.** `profile.py` covers exactly five resource types with flat field sets chosen for what an extraction model can plausibly recover from text.
- **The synthetic→real transfer gap is severe and now measured.** See above. The model as trained is **not deployable on real clinical notes**.
- **ELMTEX is published case reports, not EHR notes.** Real human clinical writing, which is the point, but more narrative and more complete than a working note. It is evidence about distribution shift, not proof of production readiness.
- **ELMTEX scores are name-level.** Its labels are free-text strings with no clinical status, dates or codes, so those numbers are not comparable in kind to the synthetic-split numbers, which match structured fields.
- **`AllergyIntolerance` rests on a thin corpus.** Synthea emits one allergy per patient — 634 encounters, 165 surviving labelled rows corpus-wide, split 105 train / 60 eval. Both halves are thin and the F1 must be read with its n.

## Reproduce it

```bash
make install
make data          # filter + patient-level split
make train         # configs/train_lora_bf16_8b.yaml
make eval
```

Serving does **not** require merging the adapter. vLLM loads a PEFT adapter directly, which also lets base and fine-tuned models be served from one process — so the A/B comparison shares a server, sampling settings and parser:

```bash
vllm serve $BASE --served-model-name qwen3-8b-base \
  --enable-lora --max-lora-rank 16 \
  --lora-modules fhir-lora=outputs/adapters/lora-bf16-8b/checkpoint-553 \
  --port 8004 --max-model-len 16384
```

`scripts/merge_adapter.py` remains for single-model deployment, but is not needed for evaluation.

## v2: the fix

The diagnosis predicted a fix, so it was tested. v2 is the same recipe — same rank, alpha, learning rate, schedule, one epoch — trained on **13,593 real ELMTEX reports** instead of Synthea-derived synthetic notes, with the split verified patient-disjoint from the test set.

| ELMTEX, conditions+procedures | micro | macro | schema validity | omitted/note |
|---|---:|---:|---:|---:|
| v1 (synthetic training) | 0.208 | 0.173 | 0.639 | 8.45 |
| base 0-shot, no fine-tune | 0.445 | 0.434 | 0.649 | 5.56 |
| **v2 (real training)** | **0.706** | **0.684** | **1.000** | **2.88** |

The repetition loop is gone entirely — 16/16 reports terminate, at a median of 412 tokens, *more* concise than the base model's 623. Schema validity reaches **1.000**: every one of 599 predictions parsed and validated.

### How much real data does it take?

Effective batch is 32, so each step is 32 examples. Evaluated on the same held-out ELMTEX set:

| v2 checkpoint | real examples seen | micro | macro | schema validity | omitted/note |
|---|---:|---:|---:|---:|---:|
| base, no fine-tune | 0 | 0.445 | 0.434 | 0.649 | 5.56 |
| step 100 | 3,200 | 0.666 | 0.639 | 0.992 | 3.47 |
| step 200 | 6,400 | 0.703 | 0.678 | 0.998 | 3.04 |
| step 300 | 9,600 | 0.708 | 0.686 | 1.000 | 3.00 |
| step 425 (full epoch) | 13,600 | 0.706 | 0.684 | 1.000 | 2.88 |

**3,200 real examples are enough to beat the un-finetuned base model by 50% relative, and the curve saturates by 9,600.** The full epoch (13,600) scores 0.002 *below* step 300 — noise, not improvement. The last third of the run bought nothing.

For comparison, v1 consumed 17,696 synthetic examples to land at 0.208, below the base model it started from. **The bottleneck was never the amount of data; it was what the data was.**

Trained on ELMTEX, only conditions/medications/procedures are supervised, since ELMTEX annotates neither vitals nor allergies. A production system would want both corpora — the synthetic one teaches the LOINC vitals coding that the demo shows the base model getting badly wrong, and the real one teaches the model to handle real prose and to stop.

## What still limits this

- **Two corpora, two conventions.** Mixing them naively would teach the model to emit empty vitals for notes that mention them. Doing it properly needs either a task marker in the prompt or a way to mark a field as "not annotated here" rather than "absent".
- **ELMTEX labels are name-level**, so v2's numbers are name-level too.
- **Neither corpus is EHR text.** Case reports are more narrative and more complete than a working clinical note.
