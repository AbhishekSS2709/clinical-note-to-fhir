import random
from fhir_extract.profile import (
    ClinicalRecord, Condition, MedicationStatement, AllergyIntolerance,
    VitalObservation, Procedure,
)
from fhir_extract.synthea import EncounterRecord
from fhir_extract.subset import select_subset, is_narratable


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


def test_allergy_only_record_is_never_empty():
    """An encounter whose only fact is an allergy must not be dropped when
    the allergy-mention roll fails; the fallback must reach allergies too."""
    allergy_only = ClinicalRecord(allergies=[AllergyIntolerance(substance_text="penicillin")])
    for seed in range(100):
        sub = select_subset(_enc(allergy_only), random.Random(seed))
        total = (len(sub.conditions) + len(sub.medications) + len(sub.allergies)
                 + len(sub.vitals) + len(sub.procedures))
        assert total > 0


# -- is_narratable ------------------------------------------------------

def test_narratable_clinical_procedures_are_kept():
    """These are things a clinician actually writes; keyword matching (not a
    blanket "(procedure)" drop) must let them through."""
    for term in ("Echocardiography (procedure)",
                 "Oxygen administration by mask (procedure)",
                 "Indirect gonioscopy (procedure)",
                 "Placing subject in prone position (procedure)"):
        assert is_narratable(term, "Procedure") is True, term


def test_administrative_procedures_are_excluded():
    for term in ("Depression screening (procedure)",
                 "Assessment of substance use (procedure)",
                 "Medication reconciliation (procedure)",
                 "Patient referral for dental care (procedure)",
                 "Oral health education (procedure)",
                 "Assessment using Morse Fall Scale (procedure)",
                 "Screening for domestic abuse (procedure)"):
        assert is_narratable(term, "Procedure") is False, term


def test_situation_tag_is_excluded_for_any_resource():
    assert is_narratable("Medication review due (situation)", "MedicationStatement") is False


def test_unknown_is_excluded_for_every_resource():
    for resource in ("Condition", "MedicationStatement", "Procedure", "AllergyIntolerance"):
        assert is_narratable("unknown", resource) is False


def test_social_determinant_findings_are_excluded():
    assert is_narratable("Stress (finding)", "Condition") is False


def test_clinical_findings_are_kept():
    assert is_narratable("Body mass index 30+ - obesity (finding)", "Condition") is True


def test_select_subset_never_selects_administrative_procedures():
    src = ClinicalRecord(
        procedures=[Procedure(code_text="Depression screening (procedure)"),
                    Procedure(code_text="Echocardiography (procedure)")],
    )
    for seed in range(30):
        sub = select_subset(_enc(src), random.Random(seed))
        texts = {p.code_text for p in sub.procedures}
        assert "Depression screening (procedure)" not in texts
