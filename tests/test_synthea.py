import json
from pathlib import Path
from fhir_extract.synthea import parse_bundle, EncounterRecord

FIXTURE = Path(__file__).parent / "fixtures" / "bundle_sample.json"
BP_COMPONENT_FIXTURE = Path(__file__).parent / "fixtures" / "bundle_bp_component.json"


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


def test_observation_with_multiple_vital_codings_yields_at_most_one_vital():
    """Ruling G: an Observation whose code.coding[] contains more than one
    LOINC that maps into VITAL_LOINC must contribute at most one
    VitalObservation, not one per matching coding."""
    bundle = {
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1",
                           "birthDate": "1980-01-01", "gender": "male"}},
            {"resource": {"resourceType": "Encounter", "id": "e1",
                           "period": {"start": "2020-01-01T00:00:00Z"}}},
            {"resource": {
                "resourceType": "Observation",
                "id": "o1",
                "code": {"coding": [
                    {"system": "http://loinc.org", "code": "8480-6",
                     "display": "Systolic blood pressure"},
                    {"system": "http://loinc.org", "code": "8462-4",
                     "display": "Diastolic blood pressure"},
                ]},
                "valueQuantity": {"value": 120, "unit": "mm[Hg]"},
                "encounter": {"reference": "Encounter/e1"},
            }},
        ]
    }
    encounters = parse_bundle(bundle)
    assert len(encounters) == 1
    assert len(encounters[0].record.vitals) == 1


def test_resource_with_no_code_text_is_skipped_not_labelled_unknown():
    """The parser's own '.get(..., "unknown")' fallback must never leak into
    a label -- a Condition/Procedure/Medication/Allergy with no code.text is
    dropped entirely rather than recorded as the placeholder "unknown"."""
    bundle = {
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1",
                           "birthDate": "1980-01-01", "gender": "male"}},
            {"resource": {"resourceType": "Encounter", "id": "e1",
                           "period": {"start": "2020-01-01T00:00:00Z"}}},
            {"resource": {
                "resourceType": "Condition", "id": "c1",
                "code": {},  # no text
                "encounter": {"reference": "Encounter/e1"},
            }},
            {"resource": {
                "resourceType": "Condition", "id": "c2",
                "code": {"text": "Chronic sinusitis (disorder)"},
                "encounter": {"reference": "Encounter/e1"},
            }},
            {"resource": {
                "resourceType": "MedicationStatement", "id": "m1",
                "medicationCodeableConcept": {},
                "encounter": {"reference": "Encounter/e1"},
            }},
            {"resource": {
                "resourceType": "Procedure", "id": "pr1",
                "code": {},
                "encounter": {"reference": "Encounter/e1"},
            }},
            {"resource": {
                "resourceType": "AllergyIntolerance", "id": "a1",
                "code": {},
            }},
        ]
    }
    encounters = parse_bundle(bundle)
    assert len(encounters) == 1
    rec = encounters[0].record
    assert [c.code_text for c in rec.conditions] == ["Chronic sinusitis (disorder)"]
    assert rec.medications == []
    assert rec.procedures == []
    assert rec.allergies == []


def test_blood_pressure_panel_yields_both_components():
    """Real Synthea exports BP as one Observation (85354-9, not itself in
    VITAL_LOINC) with no top-level valueQuantity, carrying systolic (8480-6)
    and diastolic (8462-4) inside component[]. Both must be extracted."""
    bundle = json.loads(BP_COMPONENT_FIXTURE.read_text(encoding="utf-8"))
    encounters = parse_bundle(bundle)
    assert len(encounters) == 1
    vitals = {v.loinc_code: v.value for v in encounters[0].record.vitals}
    assert vitals == {"8480-6": 132.0, "8462-4": 84.0}
