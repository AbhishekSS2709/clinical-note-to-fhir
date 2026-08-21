# Clinical Note → FHIR

Extracts structured FHIR R4 clinical records (conditions, medications, allergies, vitals, procedures) from free-text clinical notes, via a LoRA fine-tuned Qwen3-8B served side-by-side with its untouched base model.

Training data is manufactured by **reverse generation**: Synthea emits valid FHIR bundles, an LLM writes clinical notes from them, and the bundle subset *is* the label — correct by construction, with no human annotation and no teacher model to inherit errors from.

**The headline result is a negative one, and it is the point of the project:** the fine-tuned model reaches **micro-F1 0.996** on its synthetic test split and **0.209** on real clinical reports — *below* the un-finetuned base model's 0.445. Section [Where it fails](#where-it-fails-and-why) explains why, with the diagnostic evidence.

## Results

Every number is read from `outputs/eval/*.json`; none is typed by hand. All systems share one vLLM endpoint, so sampling, parser and server settings are identical across rows.

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
| **Qwen3-8B 0-shot (base)** | **0.445** | **0.434** | 0.649 | 5.56 |
| Qwen3-8B 5-shot | 0.379 | 0.382 | 0.848 | 6.76 |
| Qwen3-8B + LoRA bf16 | 0.209 | 0.173 | 0.639 | 8.45 |
| Qwen3-8B + QLoRA 4-bit | 0.199 | 0.179 | 0.591 | 8.53 |

## Where it fails, and why

The fine-tuned model does not merely score lower on real text — it **degenerates into unbounded repetition**. Measured on 16 ELMTEX reports with greedy decoding and identical prompts:

| Model | max_tokens | Hit token cap | Parsed | Median output tokens |
|---|---:|---:|---:|---:|
| base | 2048 | 0 | 16/16 | 623 |
| base | 4096 | 0 | 15/16 | 623 |
| fine-tuned | 2048 | **10** | 5/16 | 2048 |
| fine-tuned | 4096 | **10** | 5/16 | 4096 |

The base model terminates at a median of 623 tokens. The fine-tuned model consumes *whatever budget it is given*, repeating entries until it runs out. Doubling the budget changes nothing, and constrained decoding does not help either — the loop happens inside the JSON arrays, which the schema permits.

The raw output shows what it actually learned:

```json
{"medication_text": "100 MG/ML Hydrocortisone Injectable Powder/Injection",
 "dosage": {"dose": null, "unit": null, "route": null, "frequency": null},
 "status": "active"}
```

Synthea's verbose RxNorm display names, SNOMED `(finding)` tags, every optional field spelled out as `null`, and an `onset_date` it invents. It learned the *generator's surface form*, not the extraction task. Synthetic labels carry a median of 6 facts; real reports carry ~12, and the model never learned to terminate a long list.

**This is the argument for external validation.** In-distribution, every metric said the project was a success.

## The LoRA vs QLoRA ablation

Both runs are identical except precision — same effective batch (32), learning rate (2e-4), rank (16), target modules, epoch count and data.

| | bf16 LoRA | QLoRA 4-bit |
|---|---:|---:|
| eval loss @200 / @400 / @553 | 0.0056 / 0.0049 / **0.0048** | 0.0057 / 0.0050 / 0.0050 |
| synthetic micro-F1 | **0.996** | 0.992 |
| allergy-slice micro-F1 | **0.993** | 0.989 |
| ELMTEX micro-F1 | 0.209 | 0.199 |

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

## What a v2 would change

- Train on **real** annotated notes — ELMTEX ships a 54k-example training split (CC-BY-4.0).
- Train on denser labels; the synthetic corpus never shows a list long enough to teach termination.
- Report external validation from the first experiment, not the last.
