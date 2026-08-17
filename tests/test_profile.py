import pytest
from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, Dosage,
    AllergyIntolerance, VitalObservation, Procedure,
    validate_as_fhir, VITAL_LOINC, CLINICAL_STATUSES,
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


def test_medication_statement_uses_r4b_not_r5():
    """R5 renamed medicationCodeableConcept -> medication; guard against a silent bump."""
    rec = ClinicalRecord(medications=[MedicationStatement(medication_text="Metformin")])
    assert validate_as_fhir(rec) == []


def test_loinc_code_schema_enum_covers_exactly_the_eight_vital_codes():
    """A Literal type (not a free string) is what lets constrained decoding
    structurally rule out an invalid LOINC code; this is what makes
    schema_validity a meaningful metric."""
    schema = ClinicalRecord.model_json_schema()
    vitals_ref = schema["properties"]["vitals"]["items"]["$ref"]
    def_name = vitals_ref.rsplit("/", 1)[-1]
    loinc_schema = schema["$defs"][def_name]["properties"]["loinc_code"]
    expected = {code for code, _ in VITAL_LOINC.values()}
    assert set(loinc_schema["enum"]) == expected
    assert len(expected) == 8


def test_clinical_statuses_constant_matches_literal():
    """CLINICAL_STATUSES must mirror the ClinicalStatus Literal exactly (Ruling E)."""
    assert CLINICAL_STATUSES == frozenset({
        "active", "recurrence", "relapse", "inactive", "remission", "resolved"
    })
