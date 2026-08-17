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


def test_allergy_only_record_is_never_empty():
    """An encounter whose only fact is an allergy must not be dropped when
    the allergy-mention roll fails; the fallback must reach allergies too."""
    allergy_only = ClinicalRecord(allergies=[AllergyIntolerance(substance_text="penicillin")])
    for seed in range(100):
        sub = select_subset(_enc(allergy_only), random.Random(seed))
        total = (len(sub.conditions) + len(sub.medications) + len(sub.allergies)
                 + len(sub.vitals) + len(sub.procedures))
        assert total > 0
