# ELMTEX external evaluation — what is comparable and what is not

Date: 2026-08-20

The model is trained entirely on Synthea-derived notes written by Qwen3-8B. The
synthetic test split therefore partly measures its ability to invert a generator
it was trained to invert. ELMTEX ([arXiv:2502.05638](https://arxiv.org/abs/2502.05638),
CC-BY-4.0, [doi:10.5281/zenodo.14793810](https://doi.org/10.5281/zenodo.14793810))
supplies human-written clinical case reports drawn from PubMed Central, and is
used here as the external-validity check.

**These are published case reports, not EHR notes.** They are real human clinical
writing, which is the point, but they are more narrative and more complete than a
working clinical note. ELMTEX is evidence about distribution shift, not proof of
production readiness.

## Coverage: 3 of our 5 resource types

| our type | ELMTEX categories used | notes |
|---|---|---|
| conditions | `diagnosis`, `comorbidities` | |
| procedures | `interventional_therapy`, `diagnostic_techniques_procedures` | |
| medications | `pharmacological_therapy` | **not comparable — see below** |
| vitals | *(none)* | ELMTEX has no vitals category |
| allergies | *(none)* | ELMTEX has no allergy category |

`medical_surgical_history` is deliberately unused. It mixes past diagnoses with
past operations, so assigning it to conditions or to procedures invents gold
facts, and assigning it to both double-counts them.

## Why unscored types must be EXCLUDED, not scored as empty

ELMTEX annotates no vitals, but ELMTEX reports *do* mention vitals. Scoring the
full profile against empty gold charges a false positive for every vital the
model correctly extracted — penalising it for being right. `metrics.score()`
therefore takes an explicit `resources` restriction and every result JSON records
`scored_resources`.

## Two confounds found and corrected

Both were caught by inspecting raw predictions rather than trusting the first
numbers, which showed the fine-tuned model at micro 0.178 against the base
model's 0.399.

### 1. Truncated output (affected both systems)

`max_tokens: 1024` against reports carrying a median of 12 facts:

```
FT  finish=length  tok=1024 parsed=False
BAS finish=length  tok=1024 parsed=False
```

Truncated JSON fails to parse, returns an empty record and scores zero. This
depressed `schema_validity` to 0.591–0.646 for both systems. Corrected by raising
`max_tokens` to 2048 (`configs/ab_long.yaml`). The truncated run is retained
under `outputs/eval_elmtex_trunc/` rather than deleted.

### 2. Medication granularity (asymmetric — penalises the fine-tuned model)

Our FHIR profile separates the drug name from its dose, which is the correct
representation. ELMTEX concatenates them into one string:

```
pred 'levothyroxine'                       token_sort_ratio = 59.1  -> NO MATCH
gold 'levothyroxine 100mcg once a day'                       (counted as fp AND fn)

pred 'hydrocortisone'                      token_sort_ratio = 100.0 -> matches
gold 'hydrocortisone'
```

A correctly-extracted medication is charged as both a false positive and a false
negative whenever ELMTEX's annotator recorded a dose. The base model, which does
not know our convention, emits the whole string as `medication_text` and matches
ELMTEX exactly — so the metric **rewards the less structured output**.

This is not a quality difference and must not be reported as one. Medications are
therefore excluded from the headline ELMTEX comparison, which is scored on
`conditions,procedures`. The three-type run is kept alongside it so the effect is
visible rather than hidden.

**Rejected alternatives**, recorded because each is tempting:

- *Match medications by `partial_ratio`.* Containment scores 100 for any
  substring, so short names match almost anything. Inflates the number.
- *Strip doses from ELMTEX gold with a regex.* Breaks legitimate names that
  contain digits (`Vitamin B12`, `Factor VIII`) and introduces fresh errors on
  the gold side.
- *Change the matcher only for ELMTEX.* Tuning a metric per corpus until the
  result improves is how a comparison stops meaning anything.

## Standing limitation

Even on `conditions,procedures`, ELMTEX labels are free-text names with no
clinical status, dates or codes. An ELMTEX score is **name-level** and is not
comparable in kind to the synthetic-split numbers, which match structured fields.
Report the two separately; never in the same column.

## The real finding: degenerate repetition under distribution shift

The truncation in §2 above is not a budget problem. Measured on 16 ELMTEX
reports, greedy decoding, identical prompts:

| model | max_tokens | hit `length` | parsed | median tokens |
|---|---:|---:|---:|---:|
| base | 2048 | 0 | 16/16 | 623 |
| base | 4096 | 0 | 15/16 | 623 |
| fine-tuned | 2048 | 10 | 5/16 | 2048 |
| fine-tuned | 4096 | 10 | 5/16 | 4096 |

The base model terminates at a median of 623 tokens. The fine-tuned model
consumes **whatever budget it is given** on 10 of 16 reports, emitting the same
entries repeatedly:

```
... "medication_text": "100 MG/ML Hydrocortisone Injectable Powder/Injection",
    "dosage": {"dose": null, "unit": null, "route": null, "frequency": null},
    "status": "active"}, {"medication_text": "100 MG/ML Hydrocortisone ...
```

Neither a larger budget (4096 changes nothing) nor constrained decoding
(`response_format` json_schema: 7/16 parsed vs 8/16 unconstrained) fixes it —
the loop occurs *inside* the arrays, which the schema permits.

**Conclusion: the model as trained is not deployable on real clinical reports.**
The synthetic corpus has a median of 6 facts per label and a rigid, uniform
structure; real reports carry ~12. The model learned the surface form of the
generator — full RxNorm display names, SNOMED `(finding)` tags, every optional
field spelled out as `null`, an always-present `onset_date` it invents — but
never learned to terminate a long list.

This is the reason to run an external evaluation at all. The synthetic split
reports micro F1 0.996; on real text the same model is worse than the base model
it was fine-tuned from.

### What a v2 would change

- Train on denser labels; the current corpus never shows a list longer than a
  handful of items.
- Mix in real annotated notes — ELMTEX ships a 54k-example training split.
- Apply a repetition penalty at inference (untested here; would treat the
  symptom, not the cause).
- Report external validation alongside in-distribution numbers from the start,
  not at the end.
