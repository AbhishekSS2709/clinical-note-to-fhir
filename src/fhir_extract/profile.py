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
# Mirrors the ClinicalStatus Literal above exactly. Downstream modules must
# consume this constant rather than introspecting the Literal's __args__.
CLINICAL_STATUSES: frozenset[str] = frozenset({
    "active", "recurrence", "relapse", "inactive", "remission", "resolved"
})
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
    """Validate against real FHIR R4 (R4B) models. Empty list == valid.

    Uses the external `fhir.resources` package deliberately: schema validity
    must be judged by a validator we did not write. `fhir.resources` >= 8.0
    defaults its top-level namespace to R5, which is not wire-compatible
    with the R4 payloads this profile emits (e.g. R5 renamed
    MedicationStatement.medicationCodeableConcept to .medication). We
    import from the R4B sub-namespace to validate against R4-shaped data.
    """
    from fhir.resources.R4B.condition import Condition as FCondition
    from fhir.resources.R4B.medicationstatement import MedicationStatement as FMed
    from fhir.resources.R4B.allergyintolerance import AllergyIntolerance as FAllergy
    from fhir.resources.R4B.observation import Observation as FObs
    from fhir.resources.R4B.procedure import Procedure as FProc

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
