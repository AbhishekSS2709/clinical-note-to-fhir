import pytest

from fhir_extract.generate import encounter_matches
from fhir_extract.profile import (AllergyIntolerance, ClinicalRecord, Condition,
                                  VitalObservation)


def _record(*, vitals=False, allergies=False):
    return ClinicalRecord(
        conditions=[Condition(code_text="asthma", clinical_status="active")],
        vitals=[VitalObservation(loinc_code="8867-4", value=72, unit="/min")]
        if vitals else [],
        allergies=[AllergyIntolerance(substance_text="penicillin")]
        if allergies else [],
    )


def test_vitals_mode_selects_only_encounters_with_vitals():
    assert encounter_matches(_record(vitals=True), "vitals")
    assert not encounter_matches(_record(vitals=False), "vitals")


def test_no_vitals_mode_is_the_exact_complement_of_vitals_mode():
    # The two passes must partition the corpus: an encounter belongs to
    # exactly one of them, so the supplemental pass cannot re-draw what the
    # main pass already covered.
    for rec in (_record(vitals=True), _record(vitals=False)):
        assert encounter_matches(rec, "vitals") != encounter_matches(rec, "no_vitals")


def test_allergies_mode_ignores_vitals_presence():
    # Synthea records one allergy per patient, so this pool is tiny and must
    # not be narrowed further by whether the encounter happened to have vitals.
    assert encounter_matches(_record(allergies=True, vitals=False), "allergies")
    assert encounter_matches(_record(allergies=True, vitals=True), "allergies")
    assert not encounter_matches(_record(allergies=False, vitals=True), "allergies")


def test_unknown_mode_raises_rather_than_silently_generating_the_wrong_corpus():
    with pytest.raises(ValueError, match="encounter_filter"):
        encounter_matches(_record(vitals=True), "has_vitals")
