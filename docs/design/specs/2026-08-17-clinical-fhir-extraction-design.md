# Clinical Note → FHIR Extraction via LoRA/QLoRA — Design

**Date:** 2026-08-17
**Status:** Approved design, pending implementation plan
**Goal:** A portfolio project for an Applied AI / LLM Engineer role that demonstrates production-grade fine-tuning judgment, not notebook literacy.

---

## 1. What we are building

A small open-weight model fine-tuned to convert free-text clinical notes into schema-valid structured records conforming to a documented FHIR R4 profile, plus the full engineering apparatus around it: dataset construction, evaluation harness, baseline matrix, ablation study, and a served endpoint with measured throughput.

**One-sentence pitch:**
> Distilled clinical-note → FHIR extraction into a 7B model by generating training data *backwards* from schema-valid gold records, beating a prompted 32B at one quarter the VRAM.

### Why this project carries weight

Standard fine-tuning portfolio projects fail because the fine-tune itself is ~40 lines of `SFTTrainer`. The signal lives in six places, and this design targets all six:

1. Objective, non-negotiable metrics (schema validity is binary; you cannot fudge it)
2. A schema you did **not** author (FHIR R4 is an international standard)
3. Provably correct labels (correct by construction, not by trusting a teacher model)
4. A baseline matrix with cost and latency columns
5. An ablation study that answers real engineering questions
6. A measured serving story, not just training metrics

---

## 2. Scope lock

**In scope — exactly five FHIR R4 resource types:**

| Resource | Fields extracted |
|---|---|
| `Condition` | code (text + SNOMED/ICD-10 where present), clinicalStatus, onsetDateTime |
| `MedicationStatement` | medication text, dosage (dose, unit, route, frequency), status |
| `AllergyIntolerance` | substance, reaction manifestation, criticality |
| `Observation` (vitals only) | BP systolic/diastolic, HR, temp, RR, SpO2, weight, height — value + unit + LOINC code |
| `Procedure` | code text, performedDateTime, status |

**Explicitly out of scope:** every other FHIR resource type, references between resources, extensions, `meta`, full bundle assembly, multimodal/OCR input, RLHF or preference tuning, multi-turn interaction.

> **Scope creep is the primary failure mode of this project.** Full FHIR R4 has 150+ resource types. Expanding this list is forbidden without re-approving the design.

### Honesty requirement

We claim conformance to a **documented narrowed profile**, not full FHIR conformance. The README must say this plainly. Overclaiming FHIR compliance is exactly the kind of thing a healthcare-experienced interviewer catches instantly, and it would sink an otherwise strong project.

---

## 3. Constraints and assumptions

| Constraint | Value | Consequence |
|---|---|---|
| Compute | Office server, **intermittent** access | Everything must checkpoint and resume. No run may assume it finishes. |
| GPU | **ASSUMED 24GB** — pending `nvidia-smi` | Swap points documented in §6.4. Verify before Phase 3. |
| Budget | Zero paid API spend | All models open-weight. Fully reproducible by a third party for $0. |
| Licensing | Prefer Apache 2.0 end to end | Qwen2.5 (Apache 2.0), Synthea (Apache 2.0). No ToS encumbrance on outputs. |
| Fallback compute | Kaggle T4 ×2, 30 hr/week | Critical path (one QLoRA run + eval) must fit here. |

**Open assumption to resolve first:** GPU model and VRAM. This gates student model size and whether the note generator can be 32B or must be 14B.

---

## 4. Data pipeline

This is the highest-value and highest-risk part of the project. It is also the part that differentiates it from every other fine-tuning repo.

### 4.1 Core idea — reverse generation

Conventional approach: take real notes, have a big model label them, hope the labels are right.

**Our approach: run it backwards.**

```
Synthea  ──generates──▶  valid FHIR bundle  ──(this IS the gold label)
                               │
                               ▼
                      subset selection
                               │
                               ▼
                Qwen2.5-14B/32B writes a clinical note
                      conditioned on that subset
                               │
                               ▼
                    (note, label) training pair
```

The label is correct **by construction**. No teacher model, no labeling budget, no circular evaluation, unlimited volume.

**This technique has a name and a citation — use them.** It is the *asymmetry* principle from **SynthIE** (Josifoski et al., [arXiv:2303.04132](https://arxiv.org/abs/2303.04132)): for tasks with structured outputs, generating plausible input text *from* a target structure is far easier for an LLM than extracting the structure from text, so you run the easy direction to manufacture data for the hard one.

Citing it is strictly better than presenting it as your own trick. "I applied the SynthIE asymmetry principle to clinical FHIR extraction" reads as literature-aware engineering; the same idea unattributed reads as a hunch that happened to work. Verification found **no existing work applying reverse generation to Synthea→FHIR extraction specifically**, so the *application* is genuinely novel even though the *technique* is established. That is the ideal position: defensible method, original application.

### 4.2 Gold bundle generation

- Synthea (MITRE, Apache 2.0), FHIR R4 exporter enabled.
- `./run_synthea -p 3000 -s <SEED> --exporter.fhir.export true`
- Record the seed. The entire dataset must be regenerable from it.
- Take one *encounter* per example, not one patient.
- Collapse the encounter's resources into our narrowed profile JSON.

### 4.3 Subset selection — the critical correctness detail

**A Synthea bundle is complete. A real clinical note is not.** A note does not mention every condition in the patient's lifetime history.

If we label a note with the *full* bundle, we train the model to emit facts that are not in the text — i.e. **we would be explicitly training it to hallucinate.**

Therefore:

1. Sample a realistic subset of the bundle (current active conditions, current meds, this encounter's vitals, relevant history)
2. **That subset becomes the label**
3. Generate the note conditioned on *exactly* that subset

Subset selection policy is a first-class, tested component with documented sampling rules, not an afterthought.

> This is the single most important correctness detail in the project. Get it wrong and every downstream number is measuring the wrong thing.

### 4.4 Note generation

**Generator:** Qwen2.5-32B-Instruct-AWQ (~19–20GB 4-bit) or Qwen2.5-14B-Instruct-AWQ (~9GB), served offline-batch via vLLM.

> On a 24GB card, 32B-AWQ is tight once the KV cache is allocated. Run with `max_model_len=4096` and `gpu_memory_utilization=0.92`, or drop to 14B. Decide after `nvidia-smi`.

The generator's job is *writing*, not labeling — so mid-size is sufficient. This is precisely why the no-paid-API constraint does not damage this design.

**Diversity is the risk, not quality.** A single prompt at default temperature yields thousands of notes in one voice, and the student overfits to it. Mitigation is prompt-side:

| Axis | Values |
|---|---|
| `doc_type` | SOAP progress note, discharge summary, ED note, referral letter, telephone encounter, H&P |
| `style` | terse/abbreviation-heavy, verbose narrative, bulleted, dictated-ASR transcript |
| `noise` | clean, typos, unit inconsistency (lbs↔kg, °F↔°C), copy-forward duplication, negation-heavy |

Sampling: `temperature=0.9`, `top_p=0.95`. Prompt hard-constrains the model to mention every fact in the subset and invent no clinical facts beyond demographics and filler.

**Measure diversity, don't assume it:** report distinct-2/distinct-3 and type-token ratio over the corpus.

### 4.5 Faithfulness filter — non-optional

The generator *will* drop and add facts. A note missing a labeled fact is a poisoned training example.

For every generated pair, verify each gold value has an anchor in the note text:
- Drug/condition/substance names: fuzzy match via `rapidfuzz` + a hand-built abbreviation and synonym map (HTN↔hypertension, SOB↔shortness of breath, …)
- Numeric vitals: exact numeral presence with unit-conversion awareness

Drop or regenerate failures. **Log the drop rate — it is a reportable metric.** Expect 15–30%. This is fine; generation is free, so over-generate 2–3× the target volume.

Deterministic and model-free. Reused later as the hallucination detector in §7.

### 4.6 Splits

Split **by synthetic patient**, never by note — otherwise patient-specific phrasing leaks across the boundary.

| Split | Size | Source |
|---|---|---|
| train | ~8,000 | Synthea |
| val | 500 | Synthea |
| test-synthetic | 500 | Synthea, held-out patients |
| **test-real** | **200** | **MTSamples, hand-verified** |

**Two test sets is the point.** test-synthetic measures whether the task was learned. test-real measures whether it transfers. Reporting both, including a bad transfer number, is the honest engineering move and reads far better than a single suspiciously clean figure.

### 4.7 Real test set

> **Revised 2026-08-17.** Verification surfaced a public, manually annotated corpus that can absorb most of the hand-labeling burden.

**Primary source — ELMTEX.** 60,000 English clinical summaries drawn from PubMed Central, manually annotated across 15 structured categories, released as JSON alongside the paper ([arXiv:2502.05638](https://arxiv.org/abs/2502.05638); code and data at Fraunhofer's GitLab). Publicly available, real clinical text, human-annotated.

**Phase 0 decision gate — do this before committing:** audit ELMTEX's 15 categories against our five FHIR resource types.

- **If ≥3 map cleanly** (diagnosis→`Condition` and medical history are near-certain; medications, allergies, and vitals are the open questions) → write a deterministic ELMTEX→FHIR-profile mapper, then hand-verify ~200 mapped examples. You are now *checking a mapping* rather than reading raw notes, which cuts the labeling budget from 10–15 hours to roughly **3–5**.
- **If <3 map cleanly** → fall back to MTSamples as originally planned, full manual budget.

Either way the mapper itself is a worthwhile artifact: terminology and schema mapping is a real, paid healthcare-AI task, and having written one is independently demonstrable.

**Caveat to state in the README:** ELMTEX summaries are derived from *published case reports*, which are cleaner and more narrative than production EHR notes. So ELMTEX measures transfer to real clinical *prose*, not transfer to real EHR text. Do not overclaim. If time allows, keep a smaller MTSamples slice (~50 notes) as a third, harder test set — the gap between the two is itself an interesting number.

**Credibility cheap-trick:** re-annotate 30 examples two weeks later and report self-agreement.

**Licensing care:** confirm ELMTEX's and MTSamples' terms before use. Do **not** redistribute either raw corpus. Publish only your annotations/mappings keyed by source ID, plus a fetch script.

---

## 5. Model selection

> **Revised 2026-08-17 after verification.** The original draft specified Qwen2.5, which is superseded as of mid-2026. Verify current model rankings again at implementation time — this table has a shelf life measured in months.

| Role | Model | License |
|---|---|---|
| Student (primary) | **Qwen3-8B** | Apache 2.0 |
| Student (ablation) | **Granite 4.1 8B** (IBM) | Apache 2.0 |
| Student (cost ablation) | **Qwen3-4B** | Apache 2.0 |
| Note generator | Qwen3-14B / 32B-Instruct (AWQ) | Apache 2.0 |
| Baseline (large) | Qwen3-32B-Instruct | Apache 2.0 |
| Licensing-contrast ablation | Llama-3.1-8B-Instruct | Llama Community — **not** Apache |

**Why Qwen3-8B primary:** strongest JSON/tool-calling adherence in its size class, Apache 2.0, and it is the model the recent clinical-extraction literature is converging on, which makes your numbers comparable to published work.

**Why the 4B ablation matters more than it looks.** Recent published results show a 4B-class model reaching ~96.6% F1 on structured extraction — within ~0.35 points of its 8B counterpart at roughly half the parameters. If that reproduces here, "4B is sufficient" is a *better* engineering result than "8B works," because it halves serving cost. Chase this deliberately; a negative result is still publishable in your README.

**Why keep the Llama row:** solely to discuss the licensing difference. Demonstrating that you checked whether a model is commercially deployable is a cheap, high-signal differentiator.

---

## 6. Training

### 6.1 Stack

`transformers` + `peft` + `trl`, optionally **Unsloth** for roughly 2× single-GPU speed and lower VRAM — a reasonable call given intermittent compute.

### 6.2 Baseline configuration (24GB, QLoRA)

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  num_train_epochs: 3
  max_seq_length: 2048
  packing: false
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16      # effective batch 32
  gradient_checkpointing: true
  optim: paged_adamw_8bit
  bf16: true
  save_steps: 100                       # resumability
  seed: 42
```

### 6.3 Two decisions that matter more than the hyperparameters

**Target all linear layers, not just attention.** Attention-only LoRA consistently underperforms on structured-output tasks. This is ablation #3 precisely because most tutorials get it wrong.

**Compute loss on completion tokens only.** Mask the prompt. On extraction tasks the prompt is long and the completion is short; training on the full sequence spends most of the gradient signal on reproducing input text. This is the most commonly missed detail in fine-tuning tutorials and is ablation #4.

> API note: TRL's mechanism for this has changed across versions (`DataCollatorForCompletionOnlyLM` → `completion_only_loss` / `assistant_only_loss`). Pin the TRL version and verify masking empirically by decoding a batch's labels before trusting it.

**Packing is off.** Packing concatenates examples across boundaries and corrupts label structure for structured output.

### 6.4 Swap points if VRAM differs

| VRAM | Student | Mode | Generator |
|---|---|---|---|
| 16GB | Qwen2.5-7B | QLoRA, seq 1024, batch 1 | 14B-AWQ |
| 24GB | Qwen2.5-7B | QLoRA (baseline above) | 14B, or 32B at `max_model_len=4096` |
| 40GB+ | Qwen2.5-7B or 14B | bf16 LoRA comfortable | 32B |

### 6.5 Intermittent-compute discipline

Checkpoint every 100 steps; every entrypoint supports `--resume_from_checkpoint`. W&B run IDs are stable across resumes. All runs are config-file driven so a killed job restarts with one command and zero decisions.

This is not a workaround — it is how real training infra works, and the git history showing it is a positive signal.

---

## 7. Metrics

| Metric | Definition |
|---|---|
| **Field-level F1** (micro + macro) | Fuzzy match for text fields (normalized, abbreviation-expanded), exact for numerics/units/codes. Matching rule must be documented and unit-tested. |
| **Schema validity %** | Parses as JSON **and** validates against the FHIR profile via the `fhir.resources` package (external validator, not ours) |
| **Hallucination rate** | Predicted facts with no anchor in the source note — reuses §4.5 verifier |
| **Omission rate** | Gold facts absent from the prediction |
| **Per-resource breakdown** | Expect `AllergyIntolerance` to be hardest; the breakdown is where error analysis starts |
| **Catastrophic forgetting** | MMLU subset (~500 q) + instruction-following probe, before vs after. Report the drop honestly. |
| **Ops** | tok/s, p50/p95 latency, peak VRAM, cost per 1M notes |

---

## 8. Baselines — run these *before* training

| Baseline | Purpose |
|---|---|
| Regex / rule-based | Will do well on vitals, badly elsewhere. Include it; the honesty is the point. |
| medspaCy + scispaCy NER | Classical clinical NLP reference point |
| Qwen2.5-7B-Instruct zero-shot | **The headline delta** — same model, no tuning |
| Qwen2.5-7B-Instruct 5-shot | Fair prompting effort |
| Qwen2.5-7B 5-shot **+ constrained decoding** | Isolates fine-tuning from decoding — see below |
| Qwen2.5-32B-Instruct 5-shot | The "beat a 4× larger model" claim |

> **Rigor requirement:** comparing tuned+constrained against untuned+unconstrained conflates two variables and inflates your result. The 7B few-shot + constrained row exists specifically to prevent that. An interviewer who spots this omission will discount the whole table.

**Build the eval harness and run baselines in Phase 2, before any training.** Most people train first and evaluate after. Doing it in the correct order is visible in the git history and is a genuine seniority signal.

---

## 9. Ablation matrix — the resume core

**Must-do:**

1. **QLoRA vs bf16 LoRA** — quality delta, VRAM, wall-clock.
2. **Completion-only loss vs full-sequence loss** — expect a large gap
3. **Synthetic-only vs synthetic + real supplement** (100/0, 95/5, 90/10) — the transfer story
4. **Constrained decoding on/off at inference** — schema validity guaranteed by grammar vs learned
5. **8B vs 4B student** — the serving-cost result (see §5)

### ⚠ Ablation #1 is a replication, not a discovery

Verification turned up a published study answering this exact question in this exact domain: *"For clinical data extraction, QLoRA attains accuracy close to LoRA while requiring lower compute resources"* ([medRxiv 2025.10.21.25338506](https://pmc.ncbi.nlm.nih.gov/articles/PMC12633606/)), fine-tuning Llama-3.1-8B-Instruct on the ELMTEX corpus.

**Their findings — treat these as your expected values:**

| Measure | Result |
|---|---|
| LoRA gain over base | +10–20 points |
| QLoRA gain over base | +8–14 points (**2–4 points below LoRA**) |
| 4-bit QLoRA peak GPU RAM | ~⅔ of LoRA |
| QLoRA training time | **28–32% longer** (dequantization overhead) |

**Three consequences for this project:**

1. **Do not headline this ablation.** Presenting a known result as a discovery is the fastest way to look unread. Cite the paper and frame yours as independent replication on a different corpus and model family — which is a legitimate and respectable contribution.
2. **Use it as a bug detector.** You now know the expected answer. If your QLoRA run lands 15 points below LoRA, you have a bug, not a finding. This is worth more than the ablation itself.
3. **Novelty weight shifts to #3 and #5** — the synthetic→real transfer study and the 4B sufficiency result. Those are the ones with no published answer for this task.

**Nice-to-have:**

5. Rank sweep `r ∈ {8, 16, 32, 64}` — locate the knee
6. `target_modules`: attention-only vs attention+MLP
7. Base vs Instruct starting checkpoint

Cost estimate: one 7B QLoRA run over 8k examples ≈ **1.5–3 hours** on a 4090. Budget accordingly against intermittent access; must-do items come first.

---

## 10. Serving — promoted to must-have

> **Revised 2026-08-17. This is the most important change to the design.**
>
> The original draft listed a live endpoint as a *stretch goal*. Verification says that is backwards. The recurring finding across 2026 hiring guidance is blunt: **a non-technical hiring manager cannot tell a well-tuned model from a badly-tuned one by reading a README**, and a deployed system with a measurable benchmark consistently outperforms a superior fine-tune with no live endpoint.
>
> Your F1 table is invisible to the first human who screens you. A URL is not. **The endpoint is not the victory lap — it is the delivery mechanism for everything else.**

**Must-have:**

1. `peft.merge_and_unload()` → merged fp16 checkpoint
2. Serve on vLLM with structured output (`guided_json`, xgrammar backend)
3. Benchmark: tok/s at batch 1 / 8 / 32, p50/p95 latency, peak VRAM, notes/sec
4. Convert throughput to **cost per 1M notes** at a real GPU rental rate — a concrete number beats a ratio
5. **FastAPI + Docker endpoint**, paste a note → get FHIR JSON back
6. **A live demo page**, deployed on free-tier hosting. Paste a note, see structured output, side by side with the untuned baseline's output on the same input.

**Point 6 is the single highest-leverage item in this document.** The tuned-vs-untuned side-by-side makes model quality *visible in three seconds* to someone who will never read your ablation table. It converts your strongest work from invisible to obvious.

If the fine-tuned model cannot be hosted on a free tier, host the **4B** variant, or fall back to a recorded GIF walkthrough plus cached example outputs. A degraded demo beats no demo by a wide margin.

**Genuine stretch:** GGUF export + llama.cpp CPU run with measured degradation.

---

## 11. Repository and reproducibility

- Config-driven (YAML or Hydra), seeds fixed and recorded everywhere
- `make data` / `make train` / `make eval` / `make serve`
- Public W&B project
- HF Hub: adapter weights + model card carrying the full eval table
- HF Hub: synthetic dataset (no PHI — it is synthetic). MTSamples annotations published separately, keyed by source ID, with a fetch script.
- **README leads with the results table**, not installation instructions

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Large synthetic→real transfer gap | It is a **finding**, not a failure. Report it, run ablation #3, do error analysis. This becomes the most interesting section of the writeup. |
| Monotonous generated notes | Prompt rotation matrix, high temperature, measured and reported lexical diversity |
| Faithfulness filter drops too much | Over-generate 2–3× target volume; generation is free |
| Server access disappears | Full checkpoint/resume; critical path fits Kaggle T4 30hr/week |
| Hand-labeling stalls the project | ELMTEX mapping path (§4.7) cuts this to ~3–5 hrs. Either way: label **50 first**, get the pipeline end-to-end, expand to 200 after |
| Scope creep into more resources | Locked at five. Changing this requires re-approving this design. |
| Corpus licensing ambiguity | Verify ELMTEX and MTSamples terms; publish annotations only, never raw corpus |
| **Project reads as illegible to non-technical screeners** | **Phase 4 deployment + live tuned-vs-untuned demo (§10). This is the mitigation. It is not optional.** |
| Model table goes stale mid-project | Re-verify §5 at implementation time; the pipeline is model-agnostic by design |

---

## 13. Milestones

> **Revised 2026-08-17.** Deployment moved from Phase 6 to Phase 4, immediately after the first model that beats baseline. Rationale in §10.

| Phase | Deliverable | Verification |
|---|---|---|
| 0 | Env, Synthea, profile schema, validator, **ELMTEX coverage audit** (§4.7 gate) | 10 gold bundles validate against `fhir.resources`; ELMTEX map/fallback decision recorded |
| 1 | Note generation + faithfulness filter | 500 pairs produced; drop rate and diversity logged |
| 2 | **Eval harness + all baselines** | Baseline table complete on test-synthetic |
| 3 | First QLoRA run | Beats zero-shot baseline on field-F1 |
| 4 | **Deploy it** — vLLM + FastAPI + Docker + live tuned-vs-untuned demo | Public URL responds to a pasted note |
| 5 | Real test set (50 → 200) | Transfer number measured on test-real |
| 6 | Ablations 1–5 | Comparison table complete |
| 7 | Repo polish, model card, writeup | README results table; reproducible from clean clone |

**Why deploy at Phase 4 rather than at the end.** Two reasons, both practical. First, given intermittent server access, a project that stalls at Phase 5 still has a working public artifact — versus the original ordering, where stalling before Phase 6 leaves you with nothing shareable. Second, deploying early forces the inference path to stay working, so serving bugs surface while you still have context on the training code rather than weeks later.

**Minimum shippable project = Phases 0–4.** Everything after is depth. Build to that line first.

---

## 14. Draft resume bullets

> **Clinical Note → FHIR Structured Extraction** *(live demo · GitHub · model card)* — Fine-tuned Qwen3-8B (QLoRA, 4-bit NF4) to extract FHIR R4 clinical records from free-text notes. Built the training corpus by **reverse generation**, applying SynthIE's asymmetry principle to synthesize notes from schema-valid Synthea bundles so labels were correct by construction — eliminating labeling cost and circular evaluation. Achieved **X% field-F1 / Y% schema validity**, exceeding a prompted 32B baseline at ¼ the VRAM, deployed on vLLM with grammar-constrained decoding at **Z tok/s** (p95 **N** ms) on a single 24GB GPU.

> Measured synthetic→real transfer on a human-verified corpus and published the error analysis; ran controlled ablations on QLoRA vs bf16 LoRA, completion-only loss masking, LoRA rank, and 8B vs 4B student size; quantified catastrophic forgetting on MMLU. Fully open-weight Apache-2.0 pipeline, reproducible end to end at zero API cost.

**Rules for these bullets:**
- **Lead with the live demo link.** Per §10 and §16, it is what gets the rest of the bullet read.
- Fill X/Y/Z/N *only* with measured numbers. Placeholder or rounded-up figures are a fireable offense in an interview.
- Report the transfer number even if it is bad. "Measured and explained a gap" is a stronger claim than a suspiciously clean number, and the follow-up question is one you will want to be asked.

---

## 15. Open questions

1. **`nvidia-smi` output from the office server** — blocks final model-size decisions (§6.4). Resolve before Phase 3.
2. **ELMTEX category → FHIR resource coverage audit** — decides the §4.7 branch and your labeling budget. Resolve in Phase 0.
3. ELMTEX and MTSamples terms of use — confirm before Phase 5.
4. Re-verify the §5 model table at implementation time — it will drift.
5. `git init` on the project directory — the commit history is part of the deliverable.

---

## 16. Honest assessment — does this actually help the career goal?

Written after verification, because the answer is *yes, with two real caveats* rather than an unqualified yes.

### What holds up

- **Reverse generation is sound and citable** — it is SynthIE's asymmetry principle (§4.1), and no published work applies it to Synthea→FHIR extraction. Defensible method, original application.
- **Synthea genuinely produces FHIR R4 records with known ground truth.** The core premise is real.
- **The technical estimates were accurate.** The predicted "QLoRA ~30% slower, small quality cost" matched the published clinical result almost exactly (28–32% slower, 2–4 F1 points). The design rests on correct numbers.
- **Clinical structured extraction is an actively published, actively hired area** — this is not a toy domain.

### Caveat 1 — fine-tuning is not the highest-demand skill signal

2026 hiring guidance consistently reports that **RAG and agent projects appear in more job descriptions than fine-tuning**, and that fine-tuning carries a specific portfolio risk: its quality is illegible to a non-technical screener.

This does **not** mean don't do it. Fine-tuning signals deeper ML fluency than RAG, and depth is what distinguishes you in a field where everyone has shipped a RAG chatbot. But it means:

- **Deployment is mandatory, not optional** (§10). This is the entire mitigation for illegibility.
- **The strongest portfolio is one depth project plus one breadth project.** This fine-tune is the depth project. Pair it with a RAG or agent system — ideally one that *consumes this model*, e.g. an agent that ingests notes through your extraction endpoint and answers questions over the structured output. That reuses the work instead of doubling it.
- **Articulate why you chose fine-tuning over RAG.** Guidance is explicit that the *reasoning* impresses more than the choice. Here the reasoning is genuinely strong: this task needs a fixed output schema and cheap high-volume inference, not fresh retrieved knowledge. RAG is the wrong tool. Say so in the README — that paragraph is worth as much as an ablation.

### Caveat 2 — the ablation you asked about is already published

"LoRA vs QLoRA, which is better" has a published answer for clinical extraction (§9). Run it as replication and a bug-detector, cite the paper, and do not headline it. Your novel contributions are the reverse-generation pipeline, the synthetic→real transfer study, and the 4B-sufficiency result.

### What the project is actually worth

For an Applied AI / LLM Engineer role, executed to Phase 4 with a live demo, this sits **well above** the median portfolio project — because of provable label correctness, an externally-defined schema, an honest transfer study, and a measured serving story, not because it involves fine-tuning.

Executed to Phase 3 with no deployment, it is roughly *average*, regardless of how good the F1 number is.

**The delta between those two outcomes is Phase 4. Do not skip it.**
