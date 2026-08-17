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
