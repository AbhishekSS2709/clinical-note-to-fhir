"""Parse Synthea FHIR R4 bundles into per-encounter clinical records."""
import json
from datetime import date
from pathlib import Path
from typing import Iterator, Optional
from pydantic import BaseModel

from .profile import (
    ClinicalRecord, Condition, MedicationStatement, Dosage,
    AllergyIntolerance, VitalObservation, Procedure, VITAL_LOINC,
    CLINICAL_STATUSES,
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
        code_text = c.get("code", {}).get("text")
        if not code_text:
            # No usable code text; skip rather than leak our own "unknown"
            # fallback into a label.
            continue
        status = "active"
        for coding in c.get("clinicalStatus", {}).get("coding", []):
            status = coding.get("code", "active")
        rec.conditions.append(Condition(
            code_text=code_text,
            clinical_status=status if status in CLINICAL_STATUSES else "active",
            onset_date=_day(c.get("onsetDateTime")),
        ))

    for m in _entries(bundle, "MedicationRequest") + _entries(bundle, "MedicationStatement"):
        rec = bucket(_ref_id(m.get("encounter")))
        if rec is None:
            continue
        medication_text = m.get("medicationCodeableConcept", {}).get("text")
        if not medication_text:
            continue
        rec.medications.append(MedicationStatement(
            medication_text=medication_text,
            dosage=Dosage(),
            status="active",
        ))

    for a in _entries(bundle, "AllergyIntolerance"):
        # Synthea does not attach allergies to an encounter; assign to the earliest.
        if not by_encounter:
            continue
        substance_text = a.get("code", {}).get("text")
        if not substance_text:
            continue
        first = sorted(by_encounter)[0]
        by_encounter[first].allergies.append(AllergyIntolerance(
            substance_text=substance_text,
            manifestation=[r.get("manifestation", [{}])[0].get("text", "")
                           for r in a.get("reaction", []) if r.get("manifestation")],
            criticality=a.get("criticality"),
        ))

    for o in _entries(bundle, "Observation"):
        rec = bucket(_ref_id(o.get("encounter")))
        if rec is None:
            continue
        if o.get("component"):
            # Panel observations (e.g. blood pressure 85354-9) carry each
            # vital inside component[] rather than a top-level valueQuantity.
            for comp in o["component"]:
                if "valueQuantity" not in comp:
                    continue
                for coding in comp.get("code", {}).get("coding", []):
                    key = _LOINC_TO_KEY.get(coding.get("code", ""))
                    if key is None:
                        continue
                    rec.vitals.append(VitalObservation(
                        loinc_code=coding["code"],
                        display=VITAL_LOINC[key][1],
                        value=float(comp["valueQuantity"]["value"]),
                        unit=comp["valueQuantity"].get("unit", ""),
                    ))
                    break  # Ruling G: at most one vital per coding.
            continue
        if "valueQuantity" not in o:
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
            break  # Ruling G: at most one vital per Observation resource.

    for p in _entries(bundle, "Procedure"):
        rec = bucket(_ref_id(p.get("encounter")))
        if rec is None:
            continue
        code_text = p.get("code", {}).get("text")
        if not code_text:
            continue
        rec.procedures.append(Procedure(
            code_text=code_text,
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
