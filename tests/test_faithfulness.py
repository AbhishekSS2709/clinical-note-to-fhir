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

def test_empty_term_is_not_anchored():
    """Fail closed: an unverifiable fact must never count as verified."""
    rec = ClinicalRecord(conditions=[Condition(code_text="", clinical_status="active")])
    assert unanchored_facts("Patient is well.", rec) != []

def test_whitespace_term_is_not_anchored():
    rec = ClinicalRecord(conditions=[Condition(code_text="   ", clinical_status="active")])
    assert unanchored_facts("Patient is well.", rec) != []
