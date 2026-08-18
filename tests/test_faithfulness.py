from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, Procedure, VitalObservation,
)
from fhir_extract.faithfulness import (
    unanchored_facts, is_faithful, invented_facts, is_clean, normalise,
    discriminators_conflict,
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


# -- discriminator guard on _text_anchored --------------------------------
# faithfulness.py's _text_anchored uses partial_ratio at a threshold (82)
# lower than metrics.py's token_sort_ratio threshold (88), so it is the more
# permissive of the two matchers and needs the same discriminator guard, or
# a mislabelled pair can slip past faithfulness checking straight into
# training data.

def test_digit_discriminator_blocks_type_1_vs_type_2():
    rec = ClinicalRecord(conditions=[Condition(code_text="Type 2 diabetes mellitus",
                                               clinical_status="active")])
    assert is_faithful("Patient has Type 1 diabetes mellitus.", rec) is False

def test_polarity_discriminator_blocks_hyper_vs_hypo():
    rec = ClinicalRecord(conditions=[Condition(code_text="Hyperglycemia",
                                               clinical_status="active")])
    assert is_faithful("Patient found to have hypoglycemia.", rec) is False

def test_laterality_discriminator_blocks_left_vs_right():
    rec = ClinicalRecord(procedures=[Procedure(code_text="left knee replacement")])
    assert is_faithful("Patient underwent right knee replacement.", rec) is False

def test_discriminators_conflict_direct():
    assert discriminators_conflict("type 1 diabetes", "type 2 diabetes") is True
    assert discriminators_conflict("hyperglycemia", "hypoglycemia") is True
    assert discriminators_conflict("left knee", "right knee") is True
    assert discriminators_conflict("acute sinusitis", "chronic sinusitis") is True
    assert discriminators_conflict("asthma", "asthma") is False
    assert discriminators_conflict("chronic sinusitis", "chronic sinusitis disorder") is False

def test_long_note_with_unrelated_numbers_does_not_produce_spurious_conflict():
    """A term with no digits ("Anemia", fuzzily matched against the note's
    "anaemia") must still anchor via fuzzy match in a long, realistic note
    full of unrelated numbers (dates, doses, vitals). Those numbers must not
    be compared against the term just because they appear somewhere in the
    note -- a naive term-vs-whole-note discriminator check would see the
    term's empty digit set differ from the note's many digits and report a
    spurious conflict on every numeric-free term in almost any real note."""
    rec = ClinicalRecord(conditions=[Condition(code_text="Anemia",
                                               clinical_status="active")])
    note = (
        "Patient is a 54 year old presenting for routine follow up seen on "
        "2021-03-15. Vitals today are BP 148/92, heart rate 78, temperature "
        "98.6, weight 210 lbs, height 68 in. Past medical history notable "
        "for hypertension, well controlled on lisinopril 10mg daily started "
        "in 2019. Last A1c was 5.4 in March. Labs revealed mild anaemia, "
        "attributed to chronic disease. Iron studies pending. Follow up in "
        "6 months, sooner if symptoms worsen. Continue current medications. "
        "Recheck labs in 3 months if not at goal. Patient reports good "
        "adherence, denies chest pain, shortness of breath, or palpitations. "
        "Physical exam otherwise unremarkable: heart regular rate and "
        "rhythm, lungs clear to auscultation bilaterally, no edema noted, "
        "abdomen soft and non-tender."
    )
    assert unanchored_facts(note, rec) == []
    assert is_faithful(note, rec) is True
