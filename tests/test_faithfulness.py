from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, VitalObservation,
)
from fhir_extract.faithfulness import (
    unanchored_facts, is_faithful, invented_facts, is_clean, normalise,
)

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

def test_shared_generic_head_word_is_not_an_anchor():
    """'Acute bronchitis' must NOT be anchored by a note about acute pharyngitis."""
    rec = ClinicalRecord(conditions=[Condition(code_text="Acute bronchitis",
                                               clinical_status="active")])
    assert is_faithful("Patient seen for acute viral pharyngitis.", rec) is False

def test_shared_drug_stem_is_not_an_anchor():
    rec = ClinicalRecord(medications=[MedicationStatement(medication_text="Insulin glargine")])
    assert is_faithful("Started on insulin lispro", rec) is False


# -- semantic-tag stripping ----------------------------------------------

def test_semantic_tag_anchors_against_bare_term_in_note():
    rec = ClinicalRecord(conditions=[Condition(code_text="Chronic sinusitis (disorder)",
                                               clinical_status="active")])
    assert unanchored_facts("Patient reports chronic sinusitis for years.", rec) == []

def test_normalise_strips_only_trailing_semantic_tag():
    assert normalise("Chronic sinusitis (disorder)") == "chronic sinusitis"
    # Mid-string parenthetical content must survive.
    assert "abc" in normalise("abc (not a tag) def")

def test_normalise_leaves_non_tag_trailing_parens_alone():
    """Only the seven documented SNOMED semantic tags are stripped -- an
    arbitrary trailing parenthetical is real content, not a tag."""
    assert "mild" in normalise("Reaction (mild)").split()


# -- invented_facts / is_clean --------------------------------------------

def test_invented_medication_absent_from_label_is_flagged():
    rec = ClinicalRecord()  # zero medications
    note = "Patient is on intermittent antibiotic therapy with amoxicillin and clavulanic acid."
    assert invented_facts(note, rec) != []
    assert is_clean(note, rec) is False

def test_invented_facts_does_not_fire_when_drug_is_in_label():
    rec = ClinicalRecord(medications=[MedicationStatement(medication_text="Amoxicillin")])
    note = "Patient is on amoxicillin."
    assert invented_facts(note, rec) == []
    assert is_clean(note, rec) is True

def test_invented_bp_numeric_absent_from_label_is_flagged():
    rec = ClinicalRecord()
    assert invented_facts("Vitals: BP 142/88, otherwise unremarkable.", rec) != []

def test_invented_bp_numeric_matching_label_is_not_flagged():
    rec = ClinicalRecord(vitals=[VitalObservation(
        loinc_code="8480-6", display="Systolic blood pressure", value=142, unit="mmHg")])
    assert invented_facts("BP 142/88 mmHg.", rec) == []

def test_is_clean_requires_both_no_missing_and_no_invented():
    rec = ClinicalRecord(conditions=[Condition(code_text="Asthma", clinical_status="active")])
    assert is_clean("Patient is well.", rec) is False  # missing
    assert is_clean("Patient has asthma and is on amoxicillin.", rec) is False  # invented
