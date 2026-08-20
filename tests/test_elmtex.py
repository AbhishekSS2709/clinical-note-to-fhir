import pytest

from fhir_extract.elmtex import ELMTEX_FIELDS, split_entries, to_record


def test_split_entries_splits_on_semicolons():
    assert split_entries("Asthma; Hypertension; Diabetes") == \
        ["Asthma", "Hypertension", "Diabetes"]


def test_split_entries_treats_na_as_empty():
    # "N/A" is ELMTEX's own null marker; kept as a string it would become a
    # phantom gold fact that no model can ever match.
    for na in ("N/A", "n/a", " N/A ", ""):
        assert split_entries(na) == []


def test_split_entries_drops_blank_fragments():
    assert split_entries("Asthma;; Hypertension ;") == ["Asthma", "Hypertension"]


def test_to_record_maps_the_three_covered_types():
    rec = to_record({
        "diagnosis": "Thymoma; Adrenal insufficiency",
        "comorbidities": "Hypothyroidism",
        "pharmacological_therapy": "Levothyroxine 100mcg once a day",
        "interventional_therapy": "Thymoma resection",
        "diagnostic_techniques_procedures": "Chest X-ray; MRI scan",
    })
    assert [c.code_text for c in rec.conditions] == \
        ["Thymoma", "Adrenal insufficiency", "Hypothyroidism"]
    assert [m.medication_text for m in rec.medications] == \
        ["Levothyroxine 100mcg once a day"]
    assert [p.code_text for p in rec.procedures] == \
        ["Thymoma resection", "Chest X-ray", "MRI scan"]


def test_to_record_leaves_uncovered_types_empty():
    # ELMTEX annotates neither vitals nor allergies. They must stay empty AND
    # be excluded from scoring -- the reports do mention vitals, so scoring
    # them against empty gold would punish a correct extraction.
    rec = to_record({"diagnosis": "Asthma"})
    assert rec.vitals == [] and rec.allergies == []


def test_medical_surgical_history_is_excluded_as_ambiguous():
    # It mixes past conditions and past operations; assigning it to either
    # type invents gold facts, and to both double-counts.
    assert "medical_surgical_history" not in sum(ELMTEX_FIELDS.values(), [])


def test_to_record_tolerates_missing_fields():
    assert to_record({}).conditions == []
