# Clinical Note → FHIR Extraction — Implementation Plan (Phases 0–4)


**Goal:** Ship a deployed clinical-note → FHIR extraction service backed by a QLoRA fine-tune whose training labels are correct by construction.

**Architecture:** Synthea emits schema-valid FHIR bundles; a subset-selection policy picks the facts a realistic note would mention and *that subset becomes the label*; an open-weight LLM writes the note from the subset (SynthIE asymmetry principle); a deterministic faithfulness filter discards pairs where the note lost a fact. The resulting corpus trains a QLoRA adapter on Qwen3-8B, which is merged and served on vLLM behind FastAPI with a public tuned-vs-untuned demo.

**Tech Stack:** Python 3.11, Synthea (Java), `fhir.resources`, pydantic v2, vLLM, transformers + peft + trl, bitsandbytes, rapidfuzz, FastAPI, Docker, Weights & Biases.

**Spec:** `docs/design/specs/2026-08-17-clinical-fhir-extraction-design.md`

---

## Global Constraints

- **Python 3.11.0** — confirmed present on the dev machine.
- **Platform split (verified 2026-08-17):** the dev machine is Windows 11 with **no Java and no NVIDIA GPU**. vLLM does not support native Windows. Therefore: Tasks 1–2, 4, 7–9 run on Windows; Tasks 3, 5, 6, 10–13 require **Linux** (office server, WSL2, or Kaggle). Every task below states its platform.
- **All models must be Apache 2.0 open-weight.** Zero paid API calls anywhere in the pipeline.
- **Exactly five FHIR R4 resource types:** `Condition`, `MedicationStatement`, `AllergyIntolerance`, `Observation` (vitals only), `Procedure`. Adding a sixth requires re-approving the spec.
- **Every long-running script must checkpoint and support `--resume`.** Server access is intermittent; no script may assume it runs to completion.
- **Every randomised step takes an explicit `--seed` and records it** to the output manifest. The dataset must be regenerable.
- **Config-driven:** no hyperparameter or path is hardcoded in `src/`. All live in `configs/*.yaml`.
- **Never commit raw ELMTEX or MTSamples corpora.** Annotations and fetch scripts only.
- **Claims discipline:** no number appears in README or model card unless it was produced by `make eval` and is reproducible from a recorded config + seed.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Deps, tool config |
| `Makefile` | `data`, `train`, `eval`, `serve`, `test` entrypoints |
| `configs/data.yaml` | Synthea seed, volumes, split ratios, generation params |
| `configs/train_qlora_8b.yaml` | Model id, LoRA/quant/optimiser settings |
| `configs/serve.yaml` | Serving model paths, decoding params |
| `src/fhir_extract/profile.py` | The narrowed FHIR profile: pydantic models, JSON Schema, FHIR validation |
| `src/fhir_extract/synthea.py` | Run/locate Synthea, parse bundles → per-encounter records |
| `src/fhir_extract/subset.py` | Subset-selection policy (**correctness-critical**) |
| `src/fhir_extract/prompts.py` | Prompt rotation matrix (doc_type × style × noise) |
| `src/fhir_extract/generate.py` | Subset → note, batched through vLLM, resumable |
| `src/fhir_extract/faithfulness.py` | Deterministic anchor verification |
| `src/fhir_extract/dataset.py` | Patient-level splits, manifest writing |
| `src/fhir_extract/metrics.py` | field-F1, schema validity, hallucination, omission |
| `src/fhir_extract/baselines.py` | regex / zero-shot / few-shot baselines |
| `src/fhir_extract/eval.py` | Eval harness runner |
| `src/fhir_extract/train.py` | QLoRA training entrypoint, resumable |
| `src/fhir_extract/serve.py` | FastAPI app |
| `scripts/audit_elmtex.py` | Phase 0 decision gate |
| `web/index.html` | Tuned-vs-untuned demo page |
| `tests/` | One test module per src module |

---

## Task 1: Project scaffold

**Platform:** Windows (local)

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `src/fhir_extract/__init__.py`, `tests/test_smoke.py`, `configs/data.yaml`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `fhir_extract`; `make test` runs pytest

- [ ] **Step 1: Initialise the repo**

```bash
cd "C:/Users/User/Desktop/Fine tune"
git init
git branch -M main
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
data/raw/
data/interim/
data/processed/
outputs/
checkpoints/
wandb/
*.gguf
*.safetensors
synthea/
.env
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "fhir-extract"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "fhir.resources>=7.1.0",
    "pyyaml>=6.0",
    "rapidfuzz>=3.9",
    "orjson>=3.10",
    "typer>=0.12",
]

[project.optional-dependencies]
gpu = ["torch>=2.4", "transformers>=4.44", "peft>=0.12", "trl>=0.10",
       "bitsandbytes>=0.43", "accelerate>=0.33", "datasets>=2.20",
       "vllm>=0.6", "wandb>=0.17"]
serve = ["fastapi>=0.112", "uvicorn>=0.30"]
dev = ["pytest>=8.3", "pytest-cov>=5.0", "ruff>=0.6"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write `Makefile`**

```makefile
.PHONY: help install test lint data train eval serve

help:
	@echo "install  - install package + dev deps"
	@echo "test     - run pytest"
	@echo "data     - build the training corpus"
	@echo "train    - run QLoRA fine-tune"
	@echo "eval     - run eval harness"
	@echo "serve    - start FastAPI + vLLM"

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src tests

data:
	python -m fhir_extract.dataset --config configs/data.yaml

train:
	python -m fhir_extract.train --config configs/train_qlora_8b.yaml

eval:
	python -m fhir_extract.eval --config configs/data.yaml

serve:
	uvicorn fhir_extract.serve:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 5: Write `configs/data.yaml`**

```yaml
seed: 42
synthea:
  population: 3000
  output_dir: data/raw/synthea
generation:
  model: Qwen/Qwen3-14B-Instruct-AWQ
  temperature: 0.9
  top_p: 0.95
  max_tokens: 1024
  target_pairs: 24000        # over-generate ~3x; filter down to ~8000
  batch_size: 64
splits:
  train: 8000
  val: 500
  test_synthetic: 500
paths:
  interim: data/interim
  processed: data/processed
```

- [ ] **Step 6: Write the smoke test**

```python
# tests/test_smoke.py
def test_package_imports():
    import fhir_extract
    assert fhir_extract is not None
```

- [ ] **Step 7: Install and run tests**

Run: `pip install -e ".[dev]" && pytest -v`
Expected: PASS, 1 test

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: project scaffold, config, makefile"
```

---

## Task 2: FHIR profile schema

**Platform:** Windows (local)

**Files:**
- Create: `src/fhir_extract/profile.py`, `tests/test_profile.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ClinicalRecord` (pydantic model) with fields `conditions`, `medications`, `allergies`, `vitals`, `procedures`
  - `ClinicalRecord.model_json_schema() -> dict` — used as the grammar for constrained decoding
  - `validate_as_fhir(record: ClinicalRecord) -> list[str]` — returns validation errors, empty list means valid
  - `VITAL_LOINC: dict[str, tuple[str, str]]` mapping vital key → (LOINC code, display)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profile.py
import pytest
from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, Dosage,
    AllergyIntolerance, VitalObservation, Procedure,
    validate_as_fhir, VITAL_LOINC,
)

def test_empty_record_is_valid():
    rec = ClinicalRecord()
    assert validate_as_fhir(rec) == []

def test_condition_roundtrips_to_fhir():
    rec = ClinicalRecord(conditions=[
        Condition(code_text="Type 2 diabetes mellitus",
                  clinical_status="active", onset_date="2019-04-02")
    ])
    assert validate_as_fhir(rec) == []

def test_invalid_clinical_status_rejected():
    with pytest.raises(ValueError):
        Condition(code_text="x", clinical_status="not-a-status")

def test_vital_requires_known_loinc():
    with pytest.raises(ValueError):
        VitalObservation(loinc_code="0000-0", display="bogus", value=1.0, unit="mmHg")

def test_systolic_bp_loinc_is_correct():
    assert VITAL_LOINC["systolic_bp"][0] == "8480-6"

def test_json_schema_exposes_all_five_resources():
    schema = ClinicalRecord.model_json_schema()
    assert set(schema["properties"]) == {
        "conditions", "medications", "allergies", "vitals", "procedures"
    }

def test_medication_dosage_optional_fields():
    rec = ClinicalRecord(medications=[
        MedicationStatement(medication_text="Metformin",
                            dosage=Dosage(dose=500, unit="mg", route="oral",
                                          frequency="twice daily"),
                            status="active")
    ])
    assert validate_as_fhir(rec) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fhir_extract.profile'`

- [ ] **Step 3: Implement the profile**

```python
# src/fhir_extract/profile.py
"""Narrowed FHIR R4 profile: five resource types, flat field sets.

This is NOT full FHIR conformance. It is a documented profile derived from
FHIR R4, restricted to the fields an extraction model can plausibly recover
from a clinical note. See spec section 2.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

# LOINC codes for the eight vitals we support.
VITAL_LOINC: dict[str, tuple[str, str]] = {
    "systolic_bp":  ("8480-6",  "Systolic blood pressure"),
    "diastolic_bp": ("8462-4",  "Diastolic blood pressure"),
    "heart_rate":   ("8867-4",  "Heart rate"),
    "temperature":  ("8310-5",  "Body temperature"),
    "resp_rate":    ("9279-1",  "Respiratory rate"),
    "spo2":         ("2708-6",  "Oxygen saturation in Arterial blood"),
    "weight":       ("29463-7", "Body weight"),
    "height":       ("8302-2",  "Body height"),
}
_VALID_LOINC = {code for code, _ in VITAL_LOINC.values()}

ClinicalStatus = Literal[
    "active", "recurrence", "relapse", "inactive", "remission", "resolved"
]
MedicationStatus = Literal[
    "active", "completed", "entered-in-error", "intended",
    "stopped", "on-hold", "unknown", "not-taken",
]
ProcedureStatus = Literal[
    "preparation", "in-progress", "not-done", "on-hold",
    "stopped", "completed", "entered-in-error", "unknown",
]
Criticality = Literal["low", "high", "unable-to-assess"]


class Condition(BaseModel):
    code_text: str
    clinical_status: ClinicalStatus
    onset_date: Optional[str] = None   # ISO-8601 date


class Dosage(BaseModel):
    dose: Optional[float] = None
    unit: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None


class MedicationStatement(BaseModel):
    medication_text: str
    dosage: Dosage = Field(default_factory=Dosage)
    status: MedicationStatus = "active"


class AllergyIntolerance(BaseModel):
    substance_text: str
    manifestation: list[str] = Field(default_factory=list)
    criticality: Optional[Criticality] = None


class VitalObservation(BaseModel):
    loinc_code: str
    display: str
    value: float
    unit: str

    @field_validator("loinc_code")
    @classmethod
    def _known_loinc(cls, v: str) -> str:
        if v not in _VALID_LOINC:
            raise ValueError(f"unsupported LOINC code {v!r}; profile covers {_VALID_LOINC}")
        return v


class Procedure(BaseModel):
    code_text: str
    performed_date: Optional[str] = None
    status: ProcedureStatus = "completed"


class ClinicalRecord(BaseModel):
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[MedicationStatement] = Field(default_factory=list)
    allergies: list[AllergyIntolerance] = Field(default_factory=list)
    vitals: list[VitalObservation] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)


def _to_fhir_dicts(record: ClinicalRecord) -> list[dict]:
    """Expand the profile into real FHIR R4 resource dicts."""
    out: list[dict] = []
    for c in record.conditions:
        r = {
            "resourceType": "Condition",
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": c.clinical_status}]},
            "code": {"text": c.code_text},
            "subject": {"reference": "Patient/example"},
        }
        if c.onset_date:
            r["onsetDateTime"] = c.onset_date
        out.append(r)
    for m in record.medications:
        out.append({
            "resourceType": "MedicationStatement",
            "status": m.status,
            "medicationCodeableConcept": {"text": m.medication_text},
            "subject": {"reference": "Patient/example"},
        })
    for a in record.allergies:
        out.append({
            "resourceType": "AllergyIntolerance",
            "code": {"text": a.substance_text},
            "patient": {"reference": "Patient/example"},
            **({"criticality": a.criticality} if a.criticality else {}),
        })
    for v in record.vitals:
        out.append({
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org",
                                 "code": v.loinc_code, "display": v.display}]},
            "subject": {"reference": "Patient/example"},
            "valueQuantity": {"value": v.value, "unit": v.unit},
        })
    for p in record.procedures:
        r = {
            "resourceType": "Procedure",
            "status": p.status,
            "code": {"text": p.code_text},
            "subject": {"reference": "Patient/example"},
        }
        if p.performed_date:
            r["performedDateTime"] = p.performed_date
        out.append(r)
    return out


def validate_as_fhir(record: ClinicalRecord) -> list[str]:
    """Validate against real FHIR R4 models. Empty list == valid.

    Uses the external `fhir.resources` package deliberately: schema validity
    must be judged by a validator we did not write.
    """
    from fhir.resources.condition import Condition as FCondition
    from fhir.resources.medicationstatement import MedicationStatement as FMed
    from fhir.resources.allergyintolerance import AllergyIntolerance as FAllergy
    from fhir.resources.observation import Observation as FObs
    from fhir.resources.procedure import Procedure as FProc

    models = {
        "Condition": FCondition, "MedicationStatement": FMed,
        "AllergyIntolerance": FAllergy, "Observation": FObs,
        "Procedure": FProc,
    }
    errors: list[str] = []
    for res in _to_fhir_dicts(record):
        model = models[res["resourceType"]]
        try:
            model.model_validate(res)
        except Exception as exc:  # noqa: BLE001 - we surface the message verbatim
            errors.append(f"{res['resourceType']}: {exc}")
    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_profile.py -v`
Expected: PASS, 7 tests

> If `fhir.resources` rejects a resource, fix the *expansion* in `_to_fhir_dicts`, never the test. The external validator is the authority.

- [ ] **Step 5: Commit**

```bash
git add src/fhir_extract/profile.py tests/test_profile.py
git commit -m "feat: narrowed FHIR R4 profile with external validation"
```

---

## Task 3: Synthea acquisition and bundle parsing

**Platform:** Linux (or Windows once Java is installed)

**Files:**
- Create: `src/fhir_extract/synthea.py`, `tests/test_synthea.py`, `tests/fixtures/bundle_sample.json`

**Interfaces:**
- Consumes: `profile.ClinicalRecord` and its member models
- Produces:
  - `EncounterRecord` (pydantic): `patient_id: str`, `encounter_id: str`, `encounter_date: str`, `age: int`, `sex: str`, `record: ClinicalRecord`
  - `parse_bundle(bundle: dict) -> list[EncounterRecord]`
  - `iter_bundles(dir: Path) -> Iterator[dict]`

- [ ] **Step 1: Install Java and Synthea**

```bash
# Linux
sudo apt-get update && sudo apt-get install -y openjdk-17-jre-headless
git clone https://github.com/synthetichealth/synthea.git
cd synthea && ./gradlew build -x test
```

Fallback if Java is unavailable: download MITRE's pre-generated Synthea FHIR R4 sample data and skip generation. You lose seed control — record which release you used in `data/raw/MANIFEST.txt`.

- [ ] **Step 2: Generate a small bundle set and save one as a fixture**

```bash
./run_synthea -p 10 -s 42 --exporter.fhir.export true
```

Copy one output file to `tests/fixtures/bundle_sample.json`. Commit the fixture — it is synthetic, so there is no PHI concern.

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_synthea.py
import json
from pathlib import Path
from fhir_extract.synthea import parse_bundle, EncounterRecord

FIXTURE = Path(__file__).parent / "fixtures" / "bundle_sample.json"

def test_parse_bundle_returns_encounters():
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    encounters = parse_bundle(bundle)
    assert len(encounters) > 0
    assert all(isinstance(e, EncounterRecord) for e in encounters)

def test_every_encounter_has_stable_patient_id():
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    encounters = parse_bundle(bundle)
    ids = {e.patient_id for e in encounters}
    assert len(ids) == 1, "one bundle is one patient"

def test_parsed_records_validate_as_fhir():
    from fhir_extract.profile import validate_as_fhir
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for e in parse_bundle(bundle):
        assert validate_as_fhir(e.record) == []

def test_vitals_use_supported_loinc_only():
    from fhir_extract.profile import VITAL_LOINC
    valid = {c for c, _ in VITAL_LOINC.values()}
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for e in parse_bundle(bundle):
        for v in e.record.vitals:
            assert v.loinc_code in valid
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_synthea.py -v`
Expected: FAIL — module not found

- [ ] **Step 5: Implement the parser**

```python
# src/fhir_extract/synthea.py
"""Parse Synthea FHIR R4 bundles into per-encounter records."""
import json
from datetime import date
from pathlib import Path
from typing import Iterator, Optional
from pydantic import BaseModel

from .profile import (
    ClinicalRecord, Condition, MedicationStatement, Dosage,
    AllergyIntolerance, VitalObservation, Procedure, VITAL_LOINC,
)

_LOINC_TO_KEY = {code: key for key, (code, _) in VITAL_LOINC.items()}


class EncounterRecord(BaseModel):
    patient_id: str
    encounter_id: str
    encounter_date: str
    age: int
    sex: str
    record: ClinicalRecord


def iter_bundles(directory: Path) -> Iterator[dict]:
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.startswith(("hospitalInformation", "practitionerInformation")):
            continue
        yield json.loads(path.read_text(encoding="utf-8"))


def _entries(bundle: dict, resource_type: str) -> list[dict]:
    return [e["resource"] for e in bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == resource_type]


def _ref_id(ref: Optional[dict]) -> Optional[str]:
    if not ref or "reference" not in ref:
        return None
    return ref["reference"].split(":")[-1].split("/")[-1]


def _day(value: Optional[str]) -> Optional[str]:
    return value[:10] if value else None


def parse_bundle(bundle: dict) -> list[EncounterRecord]:
    patients = _entries(bundle, "Patient")
    if not patients:
        return []
    patient = patients[0]
    patient_id = patient.get("id", "unknown")
    birth = patient.get("birthDate", "1970-01-01")
    sex = patient.get("gender", "unknown")

    # Group each resource under the encounter it belongs to.
    by_encounter: dict[str, ClinicalRecord] = {}
    encounters = {e["id"]: e for e in _entries(bundle, "Encounter")}
    for enc_id in encounters:
        by_encounter[enc_id] = ClinicalRecord()

    def bucket(enc_id: Optional[str]) -> Optional[ClinicalRecord]:
        return by_encounter.get(enc_id) if enc_id else None

    for c in _entries(bundle, "Condition"):
        rec = bucket(_ref_id(c.get("encounter")))
        if rec is None:
            continue
        status = "active"
        for coding in c.get("clinicalStatus", {}).get("coding", []):
            status = coding.get("code", "active")
        rec.conditions.append(Condition(
            code_text=c.get("code", {}).get("text", "unknown"),
            clinical_status=status if status in Condition.model_fields[
                "clinical_status"].annotation.__args__ else "active",
            onset_date=_day(c.get("onsetDateTime")),
        ))

    for m in _entries(bundle, "MedicationRequest") + _entries(bundle, "MedicationStatement"):
        rec = bucket(_ref_id(m.get("encounter")))
        if rec is None:
            continue
        rec.medications.append(MedicationStatement(
            medication_text=m.get("medicationCodeableConcept", {}).get("text", "unknown"),
            dosage=Dosage(),
            status="active",
        ))

    for a in _entries(bundle, "AllergyIntolerance"):
        # Synthea does not attach allergies to an encounter; assign to the earliest.
        if not by_encounter:
            continue
        first = sorted(by_encounter)[0]
        by_encounter[first].allergies.append(AllergyIntolerance(
            substance_text=a.get("code", {}).get("text", "unknown"),
            manifestation=[r.get("manifestation", [{}])[0].get("text", "")
                           for r in a.get("reaction", []) if r.get("manifestation")],
            criticality=a.get("criticality"),
        ))

    for o in _entries(bundle, "Observation"):
        rec = bucket(_ref_id(o.get("encounter")))
        if rec is None or "valueQuantity" not in o:
            continue
        for coding in o.get("code", {}).get("coding", []):
            key = _LOINC_TO_KEY.get(coding.get("code", ""))
            if key is None:
                continue
            rec.vitals.append(VitalObservation(
                loinc_code=coding["code"],
                display=VITAL_LOINC[key][1],
                value=float(o["valueQuantity"]["value"]),
                unit=o["valueQuantity"].get("unit", ""),
            ))
            break

    for p in _entries(bundle, "Procedure"):
        rec = bucket(_ref_id(p.get("encounter")))
        if rec is None:
            continue
        rec.procedures.append(Procedure(
            code_text=p.get("code", {}).get("text", "unknown"),
            performed_date=_day((p.get("performedPeriod") or {}).get("start")
                                or p.get("performedDateTime")),
            status="completed",
        ))

    out: list[EncounterRecord] = []
    for enc_id, rec in by_encounter.items():
        if not any([rec.conditions, rec.medications, rec.allergies,
                    rec.vitals, rec.procedures]):
            continue  # skip empty encounters
        enc = encounters[enc_id]
        enc_date = _day((enc.get("period") or {}).get("start")) or "1970-01-01"
        age = max(0, date.fromisoformat(enc_date).year - date.fromisoformat(birth).year)
        out.append(EncounterRecord(
            patient_id=patient_id, encounter_id=enc_id, encounter_date=enc_date,
            age=age, sex=sex, record=rec,
        ))
    return out
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_synthea.py -v`
Expected: PASS, 4 tests

- [ ] **Step 7: Commit**

```bash
git add src/fhir_extract/synthea.py tests/test_synthea.py tests/fixtures/bundle_sample.json
git commit -m "feat: parse Synthea bundles into per-encounter clinical records"
```

---

## Task 4: Subset-selection policy

> **This is the correctness-critical task in the entire project.** If the label contains facts the note does not mention, you are training the model to hallucinate, and no metric will reveal it. See spec §4.3.

**Platform:** Windows (local)

**Files:**
- Create: `src/fhir_extract/subset.py`, `tests/test_subset.py`

**Interfaces:**
- Consumes: `synthea.EncounterRecord`, `profile.ClinicalRecord`
- Produces: `select_subset(enc: EncounterRecord, rng: random.Random) -> ClinicalRecord`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_subset.py
import random
from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, AllergyIntolerance,
    VitalObservation, Procedure,
)
from fhir_extract.synthea import EncounterRecord
from fhir_extract.subset import select_subset


def _enc(record: ClinicalRecord) -> EncounterRecord:
    return EncounterRecord(patient_id="p1", encounter_id="e1",
                           encounter_date="2020-01-01", age=50, sex="male",
                           record=record)


def _full() -> ClinicalRecord:
    return ClinicalRecord(
        conditions=[Condition(code_text=f"cond{i}", clinical_status="active")
                    for i in range(10)],
        medications=[MedicationStatement(medication_text=f"med{i}") for i in range(6)],
        allergies=[AllergyIntolerance(substance_text="penicillin")],
        vitals=[VitalObservation(loinc_code="8480-6", display="Systolic blood pressure",
                                 value=120, unit="mmHg")],
        procedures=[Procedure(code_text="appendectomy")],
    )


def test_subset_is_a_subset_of_the_source():
    rng = random.Random(0)
    src = _full()
    sub = select_subset(_enc(src), rng)
    src_conditions = {c.code_text for c in src.conditions}
    assert {c.code_text for c in sub.conditions} <= src_conditions


def test_subset_drops_something_from_a_large_record():
    rng = random.Random(0)
    src = _full()
    sub = select_subset(_enc(src), rng)
    assert len(sub.conditions) < len(src.conditions), \
        "a realistic note does not restate the entire problem list"


def test_subset_is_never_empty():
    rng = random.Random(0)
    for seed in range(50):
        sub = select_subset(_enc(_full()), random.Random(seed))
        total = (len(sub.conditions) + len(sub.medications) + len(sub.allergies)
                 + len(sub.vitals) + len(sub.procedures))
        assert total > 0


def test_vitals_are_always_kept():
    """Vitals belong to this encounter; a note recording vitals records all of them."""
    rng = random.Random(3)
    src = _full()
    sub = select_subset(_enc(src), rng)
    assert len(sub.vitals) == len(src.vitals)


def test_selection_is_deterministic_for_a_seed():
    a = select_subset(_enc(_full()), random.Random(7))
    b = select_subset(_enc(_full()), random.Random(7))
    assert a.model_dump() == b.model_dump()


def test_small_records_pass_through_intact():
    small = ClinicalRecord(conditions=[Condition(code_text="flu",
                                                 clinical_status="active")])
    sub = select_subset(_enc(small), random.Random(0))
    assert len(sub.conditions) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_subset.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the policy**

```python
# src/fhir_extract/subset.py
"""Choose which facts a realistic clinical note would actually mention.

Synthea bundles are complete patient records. Real notes are not. Labelling a
generated note with the FULL bundle would train the model to emit facts absent
from the text -- i.e. to hallucinate. So we select a plausible subset FIRST,
and that subset becomes the label. See spec section 4.3.
"""
import random
from .profile import ClinicalRecord
from .synthea import EncounterRecord

# Retention rates chosen to mimic note-writing behaviour: clinicians restate
# active problems and current meds, but rarely the full historical list.
KEEP_CONDITIONS = (0.4, 0.8)   # fraction range
KEEP_MEDICATIONS = (0.5, 1.0)
KEEP_PROCEDURES = (0.5, 1.0)
ALLERGY_MENTION_PROB = 0.7     # notes often but not always restate allergies
SMALL_RECORD_THRESHOLD = 3     # at/below this, keep everything


def _sample(items: list, rng: random.Random, rate_range: tuple[float, float]) -> list:
    if len(items) <= SMALL_RECORD_THRESHOLD:
        return list(items)
    rate = rng.uniform(*rate_range)
    k = max(1, round(len(items) * rate))
    return rng.sample(items, k)


def select_subset(enc: EncounterRecord, rng: random.Random) -> ClinicalRecord:
    src = enc.record
    sub = ClinicalRecord(
        conditions=_sample(src.conditions, rng, KEEP_CONDITIONS),
        medications=_sample(src.medications, rng, KEEP_MEDICATIONS),
        procedures=_sample(src.procedures, rng, KEEP_PROCEDURES),
        # Vitals are measured AT this encounter; if the note reports vitals it
        # reports the set. Keep all.
        vitals=list(src.vitals),
        allergies=[a for a in src.allergies if rng.random() < ALLERGY_MENTION_PROB],
    )
    if not any([sub.conditions, sub.medications, sub.allergies,
                sub.vitals, sub.procedures]):
        # Never emit an empty label; fall back to one condition or one vital.
        if src.conditions:
            sub.conditions = [src.conditions[0]]
        elif src.vitals:
            sub.vitals = list(src.vitals)
        elif src.medications:
            sub.medications = [src.medications[0]]
    return sub
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_subset.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/fhir_extract/subset.py tests/test_subset.py
git commit -m "feat: subset-selection policy so labels match note content"
```

---

## Task 5: ELMTEX coverage audit (Phase 0 decision gate)

**Platform:** Linux or Windows

**Files:**
- Create: `scripts/audit_elmtex.py`, `docs/decisions/elmtex-audit.md`

**Interfaces:**
- Consumes: `profile.ClinicalRecord`
- Produces: a written decision in `docs/decisions/elmtex-audit.md` recording how many of ELMTEX's 15 categories map to our five resource types

This task has no unit test — its deliverable is a recorded decision.

- [ ] **Step 1: Fetch ELMTEX**

Clone from the Fraunhofer GitLab referenced in [arXiv:2502.05638](https://arxiv.org/abs/2502.05638) into `data/raw/elmtex/`. Confirm the licence permits your use before proceeding. **Do not commit the corpus.**

- [ ] **Step 2: Write the audit script**

```python
# scripts/audit_elmtex.py
"""Count how many ELMTEX categories map onto our five FHIR resource types."""
import json, sys
from collections import Counter
from pathlib import Path

TARGETS = {
    "conditions": ["diagnosis", "medical history", "comorbid", "condition"],
    "medications": ["medication", "drug", "treatment", "therapy"],
    "allergies": ["allergy", "allergies", "adverse"],
    "vitals": ["vital", "blood pressure", "temperature", "heart rate"],
    "procedures": ["procedure", "surgery", "operation", "intervention"],
}

def main(path: str) -> None:
    keys = Counter()
    for f in Path(path).rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        records = data if isinstance(data, list) else [data]
        for rec in records:
            if isinstance(rec, dict):
                keys.update(rec.keys())

    print(f"Found {len(keys)} distinct annotation keys\n")
    for k, n in keys.most_common():
        print(f"  {n:6d}  {k}")

    print("\n--- Mapping to our profile ---")
    mapped = 0
    for target, needles in TARGETS.items():
        hits = [k for k in keys if any(n in k.lower() for n in needles)]
        status = "MAPS" if hits else "NO MATCH"
        if hits:
            mapped += 1
        print(f"{target:14s} {status:9s} {hits}")
    print(f"\nRESULT: {mapped}/5 resource types covered.")
    print("Decision rule: >=3 -> build the mapper; <3 -> fall back to MTSamples.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/elmtex")
```

- [ ] **Step 3: Run the audit**

Run: `python scripts/audit_elmtex.py data/raw/elmtex`
Expected: a printed count and a `RESULT: N/5` line

- [ ] **Step 4: Record the decision**

Write `docs/decisions/elmtex-audit.md` containing: the date, the raw key counts, the `N/5` result, the chosen branch (**ELMTEX mapper** if N≥3, **MTSamples fallback** otherwise), and your labelling-hours estimate under that branch.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_elmtex.py docs/decisions/elmtex-audit.md
git commit -m "chore: ELMTEX coverage audit and test-set decision"
```

---

## Task 6: Prompt matrix and note generation

**Platform:** Linux + GPU

**Files:**
- Create: `src/fhir_extract/prompts.py`, `src/fhir_extract/generate.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `profile.ClinicalRecord`, `synthea.EncounterRecord`, `subset.select_subset`
- Produces:
  - `build_prompt(enc, subset, rng) -> tuple[str, dict]` returning (prompt, variant metadata)
  - `generate_notes(config_path: str, resume: bool) -> None` writing JSONL to `data/interim/pairs.jsonl`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompts.py
import random
from fhir_extract.profile import ClinicalRecord, Condition
from fhir_extract.synthea import EncounterRecord
from fhir_extract.prompts import build_prompt, DOC_TYPES, STYLES, NOISE

def _enc():
    return EncounterRecord(patient_id="p", encounter_id="e",
                           encounter_date="2020-01-01", age=61, sex="female",
                           record=ClinicalRecord())

def test_matrix_has_expected_dimensions():
    assert len(DOC_TYPES) == 6 and len(STYLES) == 4 and len(NOISE) == 5

def test_prompt_contains_every_label_fact():
    sub = ClinicalRecord(conditions=[Condition(code_text="Acute bronchitis",
                                               clinical_status="active")])
    prompt, meta = build_prompt(_enc(), sub, random.Random(0))
    assert "Acute bronchitis" in prompt

def test_prompt_metadata_records_the_variant():
    _, meta = build_prompt(_enc(), ClinicalRecord(), random.Random(0))
    assert meta["doc_type"] in DOC_TYPES
    assert meta["style"] in STYLES
    assert meta["noise"] in NOISE

def test_variants_differ_across_seeds():
    metas = {tuple(build_prompt(_enc(), ClinicalRecord(), random.Random(s))[1].values())
             for s in range(40)}
    assert len(metas) > 5, "prompt matrix must actually vary"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the prompt matrix**

```python
# src/fhir_extract/prompts.py
"""Prompt rotation matrix. Diversity is the risk here, not quality -- a single
prompt at default temperature yields thousands of notes in one voice and the
student overfits to it. See spec section 4.4."""
import json
import random
from .profile import ClinicalRecord
from .synthea import EncounterRecord

DOC_TYPES = [
    "SOAP progress note", "discharge summary", "emergency department note",
    "referral letter", "telephone encounter note", "history and physical",
]
STYLES = [
    "terse and abbreviation-heavy, as a busy clinician types",
    "verbose narrative prose",
    "bulleted and heavily structured",
    "a dictated transcript with run-on sentences",
]
NOISE = [
    "clean and well-formatted",
    "containing a few realistic typos",
    "using inconsistent units (pounds and kilograms, Fahrenheit and Celsius)",
    "with some copy-forward duplication from a previous note",
    "with heavy use of negation for pertinent negatives",
]

_TEMPLATE = """You are writing a realistic clinical note for a training corpus.

Patient: {age}-year-old {sex}. Encounter date: {date}.

You MUST mention every one of these clinical facts somewhere in the note:
{facts}

Rules:
- Mention EVERY fact listed above. Omitting one makes the note unusable.
- Do NOT invent any additional diagnoses, medications, allergies, vital
  measurements, or procedures beyond those listed. Filler such as chief
  complaint narrative, exam prose, and disposition is encouraged.
- Write it as a {doc_type}, {style}, {noise}.
- Output ONLY the note text. No preamble, no JSON, no commentary.

Note:"""


def _facts(subset: ClinicalRecord) -> str:
    lines: list[str] = []
    for c in subset.conditions:
        onset = f" (onset {c.onset_date})" if c.onset_date else ""
        lines.append(f"- Condition: {c.code_text}, status {c.clinical_status}{onset}")
    for m in subset.medications:
        d = m.dosage
        parts = [p for p in [
            f"{d.dose}{d.unit}" if d.dose and d.unit else None,
            d.route, d.frequency] if p]
        detail = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- Medication: {m.medication_text}{detail}")
    for a in subset.allergies:
        rx = f", reaction: {', '.join(a.manifestation)}" if a.manifestation else ""
        lines.append(f"- Allergy: {a.substance_text}{rx}")
    for v in subset.vitals:
        lines.append(f"- Vital sign: {v.display} = {v.value} {v.unit}")
    for p in subset.procedures:
        when = f" on {p.performed_date}" if p.performed_date else ""
        lines.append(f"- Procedure: {p.code_text}{when}")
    return "\n".join(lines) if lines else "- (no specific findings; routine visit)"


def build_prompt(enc: EncounterRecord, subset: ClinicalRecord,
                 rng: random.Random) -> tuple[str, dict]:
    meta = {
        "doc_type": rng.choice(DOC_TYPES),
        "style": rng.choice(STYLES),
        "noise": rng.choice(NOISE),
    }
    prompt = _TEMPLATE.format(
        age=enc.age, sex=enc.sex, date=enc.encounter_date,
        facts=_facts(subset), **meta,
    )
    return prompt, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Implement resumable generation**

```python
# src/fhir_extract/generate.py
"""Generate clinical notes from label subsets. Resumable: re-running skips
encounters already present in the output file."""
import json
import random
from pathlib import Path
import typer
import yaml

from .prompts import build_prompt
from .subset import select_subset
from .synthea import iter_bundles, parse_bundle

app = typer.Typer()


def _done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["encounter_id"])
            except Exception:
                continue
    return done


@app.command()
def main(config: str = "configs/data.yaml") -> None:
    from vllm import LLM, SamplingParams

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out_path = Path(cfg["paths"]["interim"]) / "pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    typer.echo(f"Resuming: {len(done)} pairs already generated")

    # Build the work list deterministically.
    work = []
    rng = random.Random(cfg["seed"])
    for bundle in iter_bundles(Path(cfg["synthea"]["output_dir"])):
        for enc in parse_bundle(bundle):
            if enc.encounter_id in done:
                continue
            subset = select_subset(enc, rng)
            prompt, meta = build_prompt(enc, subset, rng)
            work.append((enc, subset, prompt, meta))
            if len(work) + len(done) >= cfg["generation"]["target_pairs"]:
                break
        if len(work) + len(done) >= cfg["generation"]["target_pairs"]:
            break

    typer.echo(f"Generating {len(work)} notes")
    llm = LLM(model=cfg["generation"]["model"], max_model_len=4096,
              gpu_memory_utilization=0.92)
    params = SamplingParams(
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        max_tokens=cfg["generation"]["max_tokens"],
    )

    batch = cfg["generation"]["batch_size"]
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(0, len(work), batch):
            chunk = work[i:i + batch]
            outputs = llm.generate([c[2] for c in chunk], params)
            for (enc, subset, _prompt, meta), out in zip(chunk, outputs):
                fh.write(json.dumps({
                    "patient_id": enc.patient_id,
                    "encounter_id": enc.encounter_id,
                    "note": out.outputs[0].text.strip(),
                    "label": subset.model_dump(),
                    "variant": meta,
                }) + "\n")
            fh.flush()   # checkpoint every batch
            typer.echo(f"  {min(i + batch, len(work))}/{len(work)}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Smoke-test generation on 20 encounters**

Temporarily set `target_pairs: 20` in `configs/data.yaml`, then run:
`python -m fhir_extract.generate --config configs/data.yaml`
Expected: `data/interim/pairs.jsonl` contains 20 lines. Re-run the same command; expected: it reports 20 already generated and does nothing. Restore `target_pairs`.

- [ ] **Step 7: Commit**

```bash
git add src/fhir_extract/prompts.py src/fhir_extract/generate.py tests/test_prompts.py
git commit -m "feat: prompt rotation matrix and resumable note generation"
```

---

## Task 7: Faithfulness filter

**Platform:** Windows (local)

**Files:**
- Create: `src/fhir_extract/faithfulness.py`, `tests/test_faithfulness.py`

**Interfaces:**
- Consumes: `profile.ClinicalRecord`
- Produces:
  - `unanchored_facts(note: str, record: ClinicalRecord) -> list[str]` — labels with no textual support
  - `is_faithful(note: str, record: ClinicalRecord) -> bool`

Reused in Task 9 as the hallucination detector, with arguments swapped to check *predictions* against the note.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_faithfulness.py
from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, VitalObservation,
)
from fhir_extract.faithfulness import unanchored_facts, is_faithful

def test_exact_mention_is_anchored():
    rec = ClinicalRecord(conditions=[Condition(code_text="Acute bronchitis",
                                               clinical_status="active")])
    assert unanchored_facts("Patient presents with acute bronchitis.", rec) == []

def test_missing_fact_is_flagged():
    rec = ClinicalRecord(conditions=[Condition(code_text="Acute bronchitis",
                                               clinical_status="active")])
    assert unanchored_facts("Patient is well.", rec) != []

def test_abbreviation_counts_as_an_anchor():
    rec = ClinicalRecord(conditions=[Condition(code_text="Hypertension",
                                               clinical_status="active")])
    assert unanchored_facts("Hx of HTN, controlled.", rec) == []

def test_numeric_vital_must_appear():
    rec = ClinicalRecord(vitals=[VitalObservation(
        loinc_code="8480-6", display="Systolic blood pressure",
        value=142, unit="mmHg")])
    assert unanchored_facts("BP 142/88 mmHg.", rec) == []
    assert unanchored_facts("BP normal.", rec) != []

def test_fuzzy_match_tolerates_minor_variation():
    rec = ClinicalRecord(medications=[MedicationStatement(
        medication_text="Metformin hydrochloride")])
    assert unanchored_facts("Continue metformin HCl 500mg BID.", rec) == []

def test_is_faithful_wraps_the_check():
    rec = ClinicalRecord(conditions=[Condition(code_text="Asthma",
                                               clinical_status="active")])
    assert is_faithful("Known asthma.", rec) is True
    assert is_faithful("No complaints.", rec) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_faithfulness.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the filter**

```python
# src/fhir_extract/faithfulness.py
"""Deterministic check that every labelled fact has textual support.

A generated note that dropped a fact makes its label wrong, which poisons
training. Model-free and cheap by design. See spec section 4.5.
"""
import re
from rapidfuzz import fuzz
from .profile import ClinicalRecord

FUZZ_THRESHOLD = 82

# Clinical abbreviations that count as anchors for their expansion.
ABBREVIATIONS: dict[str, list[str]] = {
    "hypertension": ["htn"],
    "diabetes mellitus": ["dm", "t2dm", "t1dm"],
    "shortness of breath": ["sob", "dyspnea"],
    "coronary artery disease": ["cad"],
    "chronic obstructive pulmonary disease": ["copd"],
    "congestive heart failure": ["chf"],
    "myocardial infarction": ["mi"],
    "urinary tract infection": ["uti"],
    "chronic kidney disease": ["ckd"],
    "atrial fibrillation": ["afib", "a-fib"],
    "gastroesophageal reflux disease": ["gerd"],
    "hydrochloride": ["hcl"],
}


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _text_anchored(term: str, note_norm: str) -> bool:
    term_norm = _normalise(term).strip()
    if not term_norm:
        return True
    # Head word of a multi-word clinical term carries most of the signal.
    if term_norm in note_norm:
        return True
    for expansion, abbrevs in ABBREVIATIONS.items():
        if expansion in term_norm and any(
            re.search(rf"\b{re.escape(a)}\b", note_norm) for a in abbrevs
        ):
            return True
    head = term_norm.split()[0]
    if len(head) >= 5 and head in note_norm:
        return True
    return fuzz.partial_ratio(term_norm, note_norm) >= FUZZ_THRESHOLD


def _number_anchored(value: float, note: str) -> bool:
    as_int = f"{value:.0f}"
    as_one_dp = f"{value:.1f}"
    return bool(re.search(rf"\b{re.escape(as_int)}\b", note)
                or re.search(rf"\b{re.escape(as_one_dp)}\b", note))


def unanchored_facts(note: str, record: ClinicalRecord) -> list[str]:
    """Return human-readable descriptions of labelled facts absent from the note."""
    note_norm = _normalise(note)
    missing: list[str] = []

    for c in record.conditions:
        if not _text_anchored(c.code_text, note_norm):
            missing.append(f"Condition: {c.code_text}")
    for m in record.medications:
        if not _text_anchored(m.medication_text, note_norm):
            missing.append(f"Medication: {m.medication_text}")
    for a in record.allergies:
        if not _text_anchored(a.substance_text, note_norm):
            missing.append(f"Allergy: {a.substance_text}")
    for v in record.vitals:
        if not _number_anchored(v.value, note):
            missing.append(f"Vital: {v.display}={v.value}")
    for p in record.procedures:
        if not _text_anchored(p.code_text, note_norm):
            missing.append(f"Procedure: {p.code_text}")
    return missing


def is_faithful(note: str, record: ClinicalRecord) -> bool:
    return not unanchored_facts(note, record)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_faithfulness.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/fhir_extract/faithfulness.py tests/test_faithfulness.py
git commit -m "feat: deterministic faithfulness filter for generated pairs"
```

---

## Task 8: Dataset assembly with patient-level splits

**Platform:** Windows (local)

**Files:**
- Create: `src/fhir_extract/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `data/interim/pairs.jsonl`, `faithfulness.is_faithful`
- Produces: `data/processed/{train,val,test_synthetic}.jsonl` plus `data/processed/manifest.json` recording seed, counts, drop rate, and diversity metrics
  - `split_by_patient(rows, ratios, seed) -> dict[str, list]`
  - `distinct_n(texts, n) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
from fhir_extract.dataset import split_by_patient, distinct_n

def _rows():
    return [{"patient_id": f"p{i // 5}", "note": f"note {i}", "label": {}}
            for i in range(50)]

def test_no_patient_appears_in_two_splits():
    splits = split_by_patient(_rows(), {"train": 30, "val": 10,
                                        "test_synthetic": 10}, seed=1)
    seen = {}
    for name, rows in splits.items():
        for r in rows:
            assert seen.setdefault(r["patient_id"], name) == name, \
                "patient leaked across splits"

def test_splits_are_deterministic_for_a_seed():
    a = split_by_patient(_rows(), {"train": 30, "val": 10, "test_synthetic": 10}, seed=1)
    b = split_by_patient(_rows(), {"train": 30, "val": 10, "test_synthetic": 10}, seed=1)
    assert [r["note"] for r in a["train"]] == [r["note"] for r in b["train"]]

def test_all_rows_are_assigned():
    splits = split_by_patient(_rows(), {"train": 30, "val": 10,
                                        "test_synthetic": 10}, seed=1)
    assert sum(len(v) for v in splits.values()) == 50

def test_distinct_n_detects_repetition():
    varied = ["the cat sat on the mat", "a dog ran through the park"]
    same = ["the cat sat on the mat"] * 2
    assert distinct_n(varied, 2) > distinct_n(same, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement dataset assembly**

```python
# src/fhir_extract/dataset.py
"""Filter generated pairs and split them by patient.

Splitting by patient rather than by note prevents patient-specific phrasing
leaking across the train/test boundary. See spec section 4.6.
"""
import json
import random
from collections import defaultdict
from pathlib import Path
import typer
import yaml

from .faithfulness import is_faithful
from .profile import ClinicalRecord

app = typer.Typer()


def distinct_n(texts: list[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams. Low values mean repetition."""
    total, unique = 0, set()
    for t in texts:
        tokens = t.lower().split()
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def split_by_patient(rows: list[dict], ratios: dict[str, int],
                     seed: int) -> dict[str, list[dict]]:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    splits: dict[str, list[dict]] = {k: [] for k in ratios}
    order = list(ratios)
    idx = 0
    for patient in patients:
        # Fill splits in order until each hits its target, then overflow to train.
        while idx < len(order) and len(splits[order[idx]]) >= ratios[order[idx]]:
            idx += 1
        target = order[idx] if idx < len(order) else order[0]
        splits[target].extend(by_patient[patient])
    return splits


@app.command()
def main(config: str = "configs/data.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    src = Path(cfg["paths"]["interim"]) / "pairs.jsonl"
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, kept = 0, []
    for line in src.open(encoding="utf-8"):
        raw += 1
        row = json.loads(line)
        record = ClinicalRecord.model_validate(row["label"])
        if is_faithful(row["note"], record):
            kept.append(row)

    drop_rate = 1 - (len(kept) / raw) if raw else 0.0
    splits = split_by_patient(kept, cfg["splits"], cfg["seed"])

    for name, rows in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    manifest = {
        "seed": cfg["seed"],
        "generation_model": cfg["generation"]["model"],
        "raw_pairs": raw,
        "kept_pairs": len(kept),
        "drop_rate": round(drop_rate, 4),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "distinct_2": round(distinct_n([r["note"] for r in kept], 2), 4),
        "distinct_3": round(distinct_n([r["note"] for r in kept], 3), 4),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Build the real dataset and inspect the manifest**

Run: `make data`
Expected: `data/processed/manifest.json` exists. **Check `drop_rate` is between 0.10 and 0.40.** Above 0.40 means the generator is losing facts — tighten the prompt in `prompts.py` before continuing. Below 0.05 means the faithfulness filter is too permissive — inspect 10 kept pairs by hand.

- [ ] **Step 6: Commit**

```bash
git add src/fhir_extract/dataset.py tests/test_dataset.py data/processed/manifest.json
git commit -m "feat: faithfulness filtering and patient-level dataset splits"
```

---

## Task 9: Metrics module

**Platform:** Windows (local)

**Files:**
- Create: `src/fhir_extract/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `profile.ClinicalRecord`, `profile.validate_as_fhir`, `faithfulness.unanchored_facts`
- Produces: `score(pred: ClinicalRecord, gold: ClinicalRecord, note: str) -> dict` returning keys `tp`, `fp`, `fn`, `schema_valid`, `hallucinated`, `omitted`, and `aggregate(list[dict]) -> dict` returning `micro_f1`, `macro_f1`, `schema_validity`, `hallucination_rate`, `omission_rate`, `per_resource`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
from fhir_extract.profile import ClinicalRecord, Condition, MedicationStatement
from fhir_extract.metrics import score, aggregate

def _rec(*conditions):
    return ClinicalRecord(conditions=[
        Condition(code_text=c, clinical_status="active") for c in conditions])

def test_perfect_prediction_scores_all_true_positives():
    gold = _rec("Asthma")
    s = score(gold, gold, "Patient has asthma.")
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 0

def test_missed_fact_is_a_false_negative():
    s = score(_rec(), _rec("Asthma"), "Patient has asthma.")
    assert s["fn"] == 1 and s["tp"] == 0

def test_invented_fact_is_a_false_positive():
    s = score(_rec("Asthma"), _rec(), "Patient is well.")
    assert s["fp"] == 1

def test_fuzzy_text_match_counts_as_correct():
    s = score(_rec("asthma"), _rec("Asthma"), "asthma")
    assert s["tp"] == 1

def test_prediction_absent_from_note_is_hallucination():
    s = score(_rec("Asthma"), _rec(), "Patient is well.")
    assert s["hallucinated"] == 1

def test_schema_validity_is_reported():
    s = score(_rec("Asthma"), _rec("Asthma"), "asthma")
    assert s["schema_valid"] is True

def test_aggregate_computes_micro_f1():
    rows = [{"tp": 1, "fp": 0, "fn": 0, "schema_valid": True,
             "hallucinated": 0, "omitted": 0, "per_resource": {}},
            {"tp": 0, "fp": 1, "fn": 1, "schema_valid": True,
             "hallucinated": 1, "omitted": 1, "per_resource": {}}]
    agg = aggregate(rows)
    assert 0.0 < agg["micro_f1"] < 1.0
    assert agg["schema_validity"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement metrics**

```python
# src/fhir_extract/metrics.py
"""Field-level scoring for extracted clinical records.

Matching rule (documented because a fuzzy metric is only credible if its rule
is stated): text fields match if normalised strings are equal OR rapidfuzz
token_sort_ratio >= 88; numeric fields must match exactly to 1 decimal place.
"""
from collections import defaultdict
from rapidfuzz import fuzz

from .faithfulness import _normalise, unanchored_facts
from .profile import ClinicalRecord, validate_as_fhir

MATCH_THRESHOLD = 88
RESOURCES = ("conditions", "medications", "allergies", "vitals", "procedures")


def _keys(record: ClinicalRecord, resource: str) -> list[str]:
    if resource == "conditions":
        return [_normalise(c.code_text).strip() for c in record.conditions]
    if resource == "medications":
        return [_normalise(m.medication_text).strip() for m in record.medications]
    if resource == "allergies":
        return [_normalise(a.substance_text).strip() for a in record.allergies]
    if resource == "vitals":
        return [f"{v.loinc_code}={v.value:.1f}" for v in record.vitals]
    if resource == "procedures":
        return [_normalise(p.code_text).strip() for p in record.procedures]
    raise ValueError(resource)


def _match(pred: list[str], gold: list[str], fuzzy: bool) -> tuple[int, int, int]:
    remaining = list(gold)
    tp = 0
    for p in pred:
        hit = None
        for g in remaining:
            if p == g or (fuzzy and fuzz.token_sort_ratio(p, g) >= MATCH_THRESHOLD):
                hit = g
                break
        if hit is not None:
            remaining.remove(hit)
            tp += 1
    return tp, len(pred) - tp, len(remaining)


def score(pred: ClinicalRecord, gold: ClinicalRecord, note: str) -> dict:
    per_resource: dict[str, dict] = {}
    tp = fp = fn = 0
    for resource in RESOURCES:
        fuzzy = resource != "vitals"
        r_tp, r_fp, r_fn = _match(_keys(pred, resource), _keys(gold, resource), fuzzy)
        per_resource[resource] = {"tp": r_tp, "fp": r_fp, "fn": r_fn}
        tp, fp, fn = tp + r_tp, fp + r_fp, fn + r_fn

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "schema_valid": validate_as_fhir(pred) == [],
        # A predicted fact with no anchor in the note is a hallucination.
        "hallucinated": len(unanchored_facts(note, pred)),
        "omitted": fn,
        "per_resource": per_resource,
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def aggregate(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)

    per_resource: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        for name, counts in r.get("per_resource", {}).items():
            for k in ("tp", "fp", "fn"):
                per_resource[name][k] += counts[k]

    resource_f1 = {n: _f1(c["tp"], c["fp"], c["fn"]) for n, c in per_resource.items()}
    n = len(rows) or 1
    return {
        "n": len(rows),
        "micro_f1": round(_f1(tp, fp, fn), 4),
        "macro_f1": round(sum(resource_f1.values()) / len(resource_f1), 4)
                    if resource_f1 else 0.0,
        "schema_validity": round(sum(bool(r["schema_valid"]) for r in rows) / n, 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in rows) / n, 4),
        "omission_rate": round(sum(r["omitted"] for r in rows) / n, 4),
        "per_resource": {k: round(v, 4) for k, v in resource_f1.items()},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/fhir_extract/metrics.py tests/test_metrics.py
git commit -m "feat: field-level F1, schema validity, hallucination metrics"
```

---

## Task 10: Eval harness and baselines

> Run this **before** any training. Building the ruler before the thing you measure is the discipline signal; it is also the only way to know your fine-tune helped.

**Platform:** Linux + GPU (zero/few-shot baselines need vLLM); the regex baseline runs anywhere

**Files:**
- Create: `src/fhir_extract/baselines.py`, `src/fhir_extract/eval.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `profile.ClinicalRecord`, `metrics.score`, `metrics.aggregate`
- Produces:
  - `regex_extract(note: str) -> ClinicalRecord`
  - `LLMBaseline(model, shots, constrained).extract_batch(notes) -> list[ClinicalRecord]`
  - `eval.main(config, split, system)` writing `outputs/eval/{system}_{split}.json`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baselines.py
from fhir_extract.baselines import regex_extract

def test_regex_extracts_blood_pressure():
    rec = regex_extract("Vitals: BP 142/88 mmHg, HR 78.")
    codes = {v.loinc_code: v.value for v in rec.vitals}
    assert codes["8480-6"] == 142 and codes["8462-4"] == 88

def test_regex_extracts_heart_rate():
    rec = regex_extract("HR 78 bpm")
    assert any(v.loinc_code == "8867-4" and v.value == 78 for v in rec.vitals)

def test_regex_extracts_temperature():
    rec = regex_extract("Temp 98.6 F")
    assert any(v.loinc_code == "8310-5" for v in rec.vitals)

def test_regex_finds_nothing_in_prose():
    rec = regex_extract("The patient feels unwell today.")
    assert rec.vitals == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baselines.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement baselines**

```python
# src/fhir_extract/baselines.py
"""Baselines. The regex baseline exists to be honest: it will do well on vitals
and badly elsewhere, and reporting that is the point.

Note the `constrained` flag on LLMBaseline. A 'few-shot + constrained decoding'
baseline is mandatory: comparing tuned+constrained against untuned+unconstrained
changes two variables at once and inflates the result. See spec section 8.
"""
import json
import re
from .profile import ClinicalRecord, VitalObservation, VITAL_LOINC

_PATTERNS = [
    (r"\bBP[:\s]+(\d{2,3})\s*/\s*(\d{2,3})", ("systolic_bp", "diastolic_bp"), "mmHg"),
    (r"\b(?:HR|heart rate|pulse)[:\s]+(\d{2,3})", ("heart_rate",), "/min"),
    (r"\b(?:RR|resp(?:iratory)? rate)[:\s]+(\d{1,2})", ("resp_rate",), "/min"),
    (r"\b(?:T|temp(?:erature)?)[:\s]+(\d{2,3}(?:\.\d)?)", ("temperature",), "F"),
    (r"\b(?:SpO2|O2 sat(?:uration)?)[:\s]+(\d{2,3})", ("spo2",), "%"),
    (r"\b(?:wt|weight)[:\s]+(\d{2,3}(?:\.\d)?)", ("weight",), "kg"),
    (r"\b(?:ht|height)[:\s]+(\d{2,3}(?:\.\d)?)", ("height",), "cm"),
]


def regex_extract(note: str) -> ClinicalRecord:
    vitals: list[VitalObservation] = []
    for pattern, keys, unit in _PATTERNS:
        for match in re.finditer(pattern, note, flags=re.IGNORECASE):
            for i, key in enumerate(keys):
                code, display = VITAL_LOINC[key]
                vitals.append(VitalObservation(
                    loinc_code=code, display=display,
                    value=float(match.group(i + 1)), unit=unit))
    return ClinicalRecord(vitals=vitals)


EXTRACT_INSTRUCTION = """Extract structured clinical data from the note below.

Return ONLY a JSON object with these five keys: conditions, medications,
allergies, vitals, procedures. Extract only facts explicitly stated in the note.

Note:
{note}

JSON:"""


class LLMBaseline:
    def __init__(self, model: str, shots: int = 0, constrained: bool = False,
                 examples: list[dict] | None = None):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.90)
        self.shots = shots
        self.examples = examples or []
        guided = None
        if constrained:
            from vllm.sampling_params import GuidedDecodingParams
            guided = GuidedDecodingParams(json=ClinicalRecord.model_json_schema())
        self.params = SamplingParams(temperature=0.0, max_tokens=1024,
                                     guided_decoding=guided)

    def _prompt(self, note: str) -> str:
        prefix = ""
        for ex in self.examples[: self.shots]:
            prefix += EXTRACT_INSTRUCTION.format(note=ex["note"])
            prefix += json.dumps(ex["label"]) + "\n\n"
        return prefix + EXTRACT_INSTRUCTION.format(note=note)

    def extract_batch(self, notes: list[str]) -> list[ClinicalRecord]:
        outputs = self.llm.generate([self._prompt(n) for n in notes], self.params)
        records = []
        for out in outputs:
            try:
                records.append(ClinicalRecord.model_validate_json(
                    out.outputs[0].text.strip()))
            except Exception:
                records.append(ClinicalRecord())  # unparseable == empty prediction
        return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_baselines.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Implement the eval runner**

```python
# src/fhir_extract/eval.py
"""Eval harness. Every number in the README comes from here."""
import json
from pathlib import Path
import typer
import yaml

from .baselines import LLMBaseline, regex_extract
from .metrics import aggregate, score
from .profile import ClinicalRecord

app = typer.Typer()


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


@app.command()
def main(
    config: str = "configs/data.yaml",
    split: str = "test_synthetic",
    system: str = "regex",
    model: str = "Qwen/Qwen3-8B",
    shots: int = 0,
    constrained: bool = False,
) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    rows = _load(Path(cfg["paths"]["processed"]) / f"{split}.jsonl")
    notes = [r["note"] for r in rows]
    golds = [ClinicalRecord.model_validate(r["label"]) for r in rows]

    if system == "regex":
        preds = [regex_extract(n) for n in notes]
    else:
        examples = _load(Path(cfg["paths"]["processed"]) / "train.jsonl")[:shots]
        preds = LLMBaseline(model, shots, constrained, examples).extract_batch(notes)

    results = aggregate([score(p, g, n) for p, g, n in zip(preds, golds, notes)])
    results["system"] = system
    results["model"] = model if system != "regex" else None
    results["shots"] = shots
    results["constrained"] = constrained
    results["split"] = split

    out = Path("outputs/eval"); out.mkdir(parents=True, exist_ok=True)
    tag = f"{system}_{shots}shot{'_constrained' if constrained else ''}_{split}"
    (out / f"{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run the full baseline matrix**

```bash
python -m fhir_extract.eval --system regex
python -m fhir_extract.eval --system llm --model Qwen/Qwen3-8B --shots 0
python -m fhir_extract.eval --system llm --model Qwen/Qwen3-8B --shots 5
python -m fhir_extract.eval --system llm --model Qwen/Qwen3-8B --shots 5 --constrained
python -m fhir_extract.eval --system llm --model Qwen/Qwen3-32B-Instruct --shots 5
```

Expected: five JSON files in `outputs/eval/`. Record the `micro_f1` of the 8B zero-shot run — **this is the number your fine-tune must beat in Task 11.**

- [ ] **Step 7: Commit**

```bash
git add src/fhir_extract/baselines.py src/fhir_extract/eval.py tests/test_baselines.py outputs/eval
git commit -m "feat: eval harness and baseline matrix (pre-training)"
```

---

## Task 11: QLoRA fine-tune

**Platform:** Linux + GPU

**Files:**
- Create: `src/fhir_extract/train.py`, `configs/train_qlora_8b.yaml`

**Interfaces:**
- Consumes: `data/processed/train.jsonl`, `data/processed/val.jsonl`
- Produces: adapter checkpoint at `outputs/adapters/qlora-8b/`

- [ ] **Step 1: Confirm the GPU before anything else**

Run: `nvidia-smi --query-gpu=name,memory.total --format=csv`
Then set `configs/train_qlora_8b.yaml` per spec §6.4. **Do not skip this** — the config below assumes 24GB.

- [ ] **Step 2: Write the training config**

```yaml
# configs/train_qlora_8b.yaml
model_id: Qwen/Qwen3-8B
output_dir: outputs/adapters/qlora-8b
seed: 42
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
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  optim: paged_adamw_8bit
  bf16: true
  save_steps: 100
  logging_steps: 10
  eval_steps: 200
```

- [ ] **Step 3: Implement training**

```python
# src/fhir_extract/train.py
"""QLoRA fine-tune. Resumable by design -- server access is intermittent."""
import json
from pathlib import Path
import typer
import yaml

app = typer.Typer()

RESPONSE_MARKER = "\nJSON:\n"


def _format(row: dict) -> str:
    from .baselines import EXTRACT_INSTRUCTION
    prompt = EXTRACT_INSTRUCTION.format(note=row["note"]).rstrip()
    return f"{prompt}\n{json.dumps(row['label'])}"


@app.command()
def main(config: str = "configs/train_qlora_8b.yaml", resume: bool = True) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    proc = Path("data/processed")

    def load(name: str) -> Dataset:
        rows = [json.loads(l) for l in (proc / f"{name}.jsonl").open(encoding="utf-8")]
        return Dataset.from_dict({"text": [_format(r) for r in rows]})

    train_ds, val_ds = load("train"), load("val")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    peft_cfg = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    t = cfg["training"]
    args = SFTConfig(
        output_dir=cfg["output_dir"], seed=cfg["seed"],
        report_to="wandb", run_name=Path(cfg["output_dir"]).name,
        save_total_limit=3, packing=False, **t,
    )

    trainer = SFTTrainer(
        model=cfg["model_id"],
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        model_init_kwargs={"quantization_config": bnb, "device_map": "auto"},
    )

    ckpts = sorted(Path(cfg["output_dir"]).glob("checkpoint-*")) if resume else []
    trainer.train(resume_from_checkpoint=bool(ckpts))
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Verify completion-only loss masking is actually active**

This is the highest-value thing to check and the easiest to get silently wrong. TRL's API for it has changed across versions — pin your version, then decode one batch's labels:

```python
batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
visible = tokenizer.decode([t for t in labels if t != -100])
print(visible)
```

Expected: **only the JSON answer**, not the note. If the note appears, the mask is not applied — fix it before spending GPU hours. Record which mechanism worked in a comment.

- [ ] **Step 5: Smoke-train on 100 examples**

Temporarily point `train.jsonl` at a 100-row slice and set `num_train_epochs: 1`. Run `make train`.
Expected: completes without OOM; a checkpoint appears in `outputs/adapters/qlora-8b/`.

- [ ] **Step 6: Verify resumability**

Kill the full training run mid-way (Ctrl-C after a checkpoint is written), then re-run `make train`.
Expected: log shows it resumed from the checkpoint rather than restarting at step 0. **This is a required verification, not optional** — the whole plan depends on it.

- [ ] **Step 7: Run the full fine-tune and evaluate**

```bash
make train
python -m fhir_extract.eval --system tuned --model outputs/adapters/qlora-8b --constrained
```

Expected: `micro_f1` **exceeds the 8B zero-shot baseline from Task 10**. If it does not, stop and debug — do not proceed to deployment with a model that lost to its own base.

- [ ] **Step 8: Commit**

```bash
git add src/fhir_extract/train.py configs/train_qlora_8b.yaml outputs/eval
git commit -m "feat: QLoRA fine-tune with completion-only loss and resume support"
```

---

## Task 12: Merge and serve

**Platform:** Linux + GPU

**Files:**
- Create: `src/fhir_extract/serve.py`, `scripts/merge_adapter.py`, `Dockerfile`, `configs/serve.yaml`

**Interfaces:**
- Consumes: `outputs/adapters/qlora-8b`, `profile.ClinicalRecord`
- Produces: `POST /extract {"note": str} -> {"tuned": {...}, "base": {...}, "latency_ms": {...}}`

- [ ] **Step 1: Write the merge script**

```python
# scripts/merge_adapter.py
"""Merge the LoRA adapter into base weights for single-model serving."""
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                             device_map="cpu")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(out_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id).save_pretrained(out_dir)
print(f"merged -> {out_dir}")
```

Run: `python scripts/merge_adapter.py Qwen/Qwen3-8B outputs/adapters/qlora-8b outputs/merged/qlora-8b`

- [ ] **Step 2: Write the FastAPI app**

```python
# src/fhir_extract/serve.py
"""Serve tuned and base models side by side.

The side-by-side response is the point: it renders model quality visible in
three seconds to someone who will never read the ablation table.
See spec section 10.
"""
import time
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .baselines import EXTRACT_INSTRUCTION
from .profile import ClinicalRecord

cfg = yaml.safe_load(Path("configs/serve.yaml").read_text(encoding="utf-8"))
app = FastAPI(title="Clinical Note to FHIR")

_engines: dict[str, object] = {}


def _engine(path: str):
    if path not in _engines:
        from vllm import LLM
        _engines[path] = LLM(model=path, max_model_len=4096,
                             gpu_memory_utilization=cfg["gpu_memory_utilization"])
    return _engines[path]


class ExtractRequest(BaseModel):
    note: str


def _run(model_path: str, note: str) -> tuple[dict, float]:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    params = SamplingParams(
        temperature=0.0, max_tokens=1024,
        guided_decoding=GuidedDecodingParams(
            json=ClinicalRecord.model_json_schema()),
    )
    start = time.perf_counter()
    out = _engine(model_path).generate(
        [EXTRACT_INSTRUCTION.format(note=note)], params)
    elapsed = (time.perf_counter() - start) * 1000
    try:
        record = ClinicalRecord.model_validate_json(out[0].outputs[0].text.strip())
    except Exception:
        record = ClinicalRecord()
    return record.model_dump(), round(elapsed, 1)


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    tuned, tuned_ms = _run(cfg["tuned_model"], req.note)
    base, base_ms = _run(cfg["base_model"], req.note)
    return {"tuned": tuned, "base": base,
            "latency_ms": {"tuned": tuned_ms, "base": base_ms}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
```

- [ ] **Step 3: Write `configs/serve.yaml`**

```yaml
tuned_model: outputs/merged/qlora-8b
base_model: Qwen/Qwen3-8B
gpu_memory_utilization: 0.45   # two engines share one GPU
```

- [ ] **Step 4: Write the Dockerfile**

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[gpu,serve]"
COPY src/ src/
COPY configs/ configs/
COPY web/ web/
EXPOSE 8000
CMD ["uvicorn", "fhir_extract.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 5: Verify the endpoint**

```bash
make serve &
curl -s -X POST localhost:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"note":"62yo M with HTN and T2DM. BP 148/92, HR 82. Continue metformin 500mg BID. NKDA."}' | python -m json.tool
```

Expected: JSON with `tuned` and `base` keys; `tuned` should contain hypertension and diabetes conditions plus metformin. **If the two look identical, the merge silently failed** — verify `outputs/merged/` is not just a copy of the base.

- [ ] **Step 6: Benchmark and record**

Measure tok/s at batch 1, 8, and 32, plus p50/p95 latency and peak VRAM. Write results to `outputs/serving_benchmark.json`. Convert to cost per 1M notes using a real GPU hourly rate and record which rate you used.

- [ ] **Step 7: Commit**

```bash
git add src/fhir_extract/serve.py scripts/merge_adapter.py Dockerfile configs/serve.yaml outputs/serving_benchmark.json
git commit -m "feat: merge adapter and serve tuned vs base with FastAPI + vLLM"
```

---

## Task 13: Demo page and README

**Platform:** any

**Files:**
- Create: `web/index.html`, `README.md`

- [ ] **Step 1: Write the demo page**

```html
<!-- web/index.html -->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clinical Note to FHIR</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;
       background:#fbfbfd;color:#111}
  textarea{width:100%;height:160px;font-family:ui-monospace,monospace;font-size:13px;
           padding:.75rem;border:1px solid #ccc;border-radius:6px}
  button{padding:.6rem 1.4rem;font-size:15px;border:0;border-radius:6px;
         background:#2b6cb0;color:#fff;cursor:pointer;margin:.75rem 0}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  .col{border:1px solid #ddd;border-radius:8px;padding:1rem;background:#fff;overflow-x:auto}
  .col h3{margin:0 0 .5rem}
  .tuned{border-color:#2f855a}.base{border-color:#c05621}
  pre{font-size:12px;white-space:pre-wrap;margin:0}
  .ms{color:#666;font-size:12px}
  @media(max-width:800px){.cols{grid-template-columns:1fr}}
</style>
</head>
<body>
<h1>Clinical Note → FHIR</h1>
<p>Paste a clinical note. The fine-tuned model and its untouched base model both
extract from it, side by side.</p>
<textarea id="note">62yo M with HTN and T2DM presents for follow-up. BP 148/92, HR 82, temp 98.4F.
Continue metformin 500mg BID and lisinopril 10mg daily. NKDA. Appendectomy 2011.</textarea>
<button onclick="run()">Extract</button>
<div class="cols">
  <div class="col tuned"><h3>Fine-tuned</h3><div class="ms" id="tms"></div><pre id="tuned"></pre></div>
  <div class="col base"><h3>Base (no fine-tune)</h3><div class="ms" id="bms"></div><pre id="base"></pre></div>
</div>
<script>
async function run(){
  document.getElementById('tuned').textContent='...';
  document.getElementById('base').textContent='...';
  const r = await fetch('/extract',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({note:document.getElementById('note').value})});
  const d = await r.json();
  document.getElementById('tuned').textContent=JSON.stringify(d.tuned,null,2);
  document.getElementById('base').textContent=JSON.stringify(d.base,null,2);
  document.getElementById('tms').textContent=d.latency_ms.tuned+' ms';
  document.getElementById('bms').textContent=d.latency_ms.base+' ms';
}
</script>
</body>
</html>
```

- [ ] **Step 2: Write the README**

Required structure, in this order:

1. **One-line description + live demo link** (first thing on the page)
2. **Results table** — every baseline from Task 10 plus the fine-tune, with micro-F1, schema validity, hallucination rate, latency, and VRAM columns
3. **Why fine-tuning and not RAG** — fixed output schema and cheap high-volume inference; retrieval solves neither. Spec §16 explains why this paragraph carries weight.
4. **How the data was built** — the reverse-generation diagram, crediting [SynthIE](https://arxiv.org/abs/2303.04132), and the subset-selection rationale from §4.3
5. **Honest limitations** — narrowed FHIR profile, not full conformance; synthetic training distribution; measured transfer gap
6. **Reproduce it** — `make install && make data && make train && make eval`

Populate the results table **only** from `outputs/eval/*.json`. No hand-typed numbers.

- [ ] **Step 3: Verify the demo end to end**

Open `http://localhost:8000/` in a browser, click Extract, confirm both panes populate and visibly differ.

- [ ] **Step 4: Deploy publicly**

Deploy to any free-tier GPU host. If none is available, deploy the 4B variant, or record a GIF walkthrough and serve cached example outputs as a static page. **A degraded demo beats no demo.** Put the URL at the top of the README.

- [ ] **Step 5: Commit**

```bash
git add web/index.html README.md
git commit -m "feat: side-by-side demo page and results-first README"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| §2 Scope lock (5 resources) | Task 2 |
| §3 Constraints, platform | Global Constraints, Task 11 Step 1 |
| §4.2 Synthea gold bundles | Task 3 |
| §4.3 Subset selection | Task 4 |
| §4.4 Note generation + prompt matrix | Task 6 |
| §4.5 Faithfulness filter | Task 7 |
| §4.6 Patient-level splits | Task 8 |
| §4.7 ELMTEX gate | Task 5 |
| §5 Model selection | Task 11 config |
| §6 Training, completion-only loss, resume | Task 11 Steps 3, 4, 6 |
| §7 Metrics | Task 9 |
| §8 Baselines before training | Task 10 |
| §10 Serving + demo | Tasks 12, 13 |
| §13 Phase ordering | Task order |

**Deferred to a follow-up plan (Phases 5–7):** real-test-set labelling and transfer measurement, ablations 1–5, catastrophic-forgetting check, model card, HF Hub publication. These depend on numbers that do not exist until Task 11 completes.

**Known gaps to resolve during execution:**
- Task 11 Step 1 (`nvidia-smi`) gates the training config. Unresolved at plan time.
- Task 5's outcome determines the Phase 5 labelling branch. Unresolved at plan time.
- TRL's completion-only-loss API varies by version; Task 11 Step 4 verifies empirically rather than assuming a call signature.
