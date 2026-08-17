import pytest

from fhir_extract.profile import ClinicalRecord, Condition, MedicationStatement, Procedure
from fhir_extract.metrics import score, aggregate

def _rec(*conditions):
    return ClinicalRecord(conditions=[
        Condition(code_text=c, clinical_status="active") for c in conditions])

def test_perfect_prediction_scores_all_true_positives():
    gold = _rec("Asthma")
    s = score(gold, gold, "Patient has asthma.")
    assert s["tp"] == 1 and s["fp"] == 0 and s["fn"] == 0

def test_missed_fact_is_a_false_negative():
    s = score(_rec(), _rec("Asthma"), "Patient has asthma.")
    assert s["fn"] == 1 and s["tp"] == 0

def test_invented_fact_is_a_false_positive():
    s = score(_rec("Asthma"), _rec(), "Patient is well.")
    assert s["fp"] == 1

def test_fuzzy_text_match_counts_as_correct():
    s = score(_rec("asthma"), _rec("Asthma"), "asthma")
    assert s["tp"] == 1

def test_prediction_absent_from_note_is_hallucination():
    s = score(_rec("Asthma"), _rec(), "Patient is well.")
    assert s["hallucinated"] == 1

def test_schema_validity_is_reported():
    s = score(_rec("Asthma"), _rec("Asthma"), "asthma")
    assert s["schema_valid"] is True

def test_aggregate_computes_micro_f1():
    rows = [{"tp": 1, "fp": 0, "fn": 0, "schema_valid": True,
             "hallucinated": 0, "omitted": 0, "per_resource": {}},
            {"tp": 0, "fp": 1, "fn": 1, "schema_valid": True,
             "hallucinated": 1, "omitted": 1, "per_resource": {}}]
    agg = aggregate(rows)
    assert 0.0 < agg["micro_f1"] < 1.0
    assert agg["schema_validity"] == 1.0

def test_digit_discriminator_blocks_false_match():
    """'Type 1' vs 'Type 2' diabetes scores 93.3 fuzzy - must NOT be a true positive."""
    gold = ClinicalRecord(conditions=[Condition(code_text="Type 2 diabetes mellitus",
                                                clinical_status="active")])
    pred = ClinicalRecord(conditions=[Condition(code_text="Type 1 diabetes mellitus",
                                                clinical_status="active")])
    s = score(pred, gold, "Type 1 diabetes mellitus")
    assert s["tp"] == 0 and s["fp"] == 1 and s["fn"] == 1

def test_laterality_discriminator_blocks_false_match():
    gold = ClinicalRecord(procedures=[Procedure(code_text="left knee replacement")])
    pred = ClinicalRecord(procedures=[Procedure(code_text="right knee replacement")])
    s = score(pred, gold, "right knee replacement")
    assert s["tp"] == 0

def test_macro_f1_ignores_absent_resource_types():
    """A perfect prediction must score macro_f1 == 1.0 even when 4 of 5 types are empty."""
    rec = ClinicalRecord(conditions=[Condition(code_text="Asthma", clinical_status="active")])
    agg = aggregate([score(rec, rec, "Asthma")])
    assert agg["macro_f1"] == 1.0

def test_best_match_not_first_match():
    """A mediocre earlier candidate must not consume a gold that matches a later pred better."""
    gold = ClinicalRecord(conditions=[
        Condition(code_text="Asthma", clinical_status="active"),
        Condition(code_text="Asthma exacerbation", clinical_status="active")])
    pred = ClinicalRecord(conditions=[
        Condition(code_text="Asthma exacerbation", clinical_status="active"),
        Condition(code_text="Asthma", clinical_status="active")])
    s = score(pred, gold, "Asthma exacerbation and Asthma")
    assert s["tp"] == 2, "both should match their exact counterparts"

@pytest.mark.parametrize("gold_text,pred_text,should_match", [
    ("Hyperglycemia", "Hypoglycemia", False),
    ("Hypothyroidism", "Hyperthyroidism", False),
    ("Hypercalcemia", "Hypocalcemia", False),
    ("Acute pancreatitis", "Chronic pancreatitis", False),
    ("Asthma", "asthma", True),
    ("Asthma", "Asthma", True),
])
def test_polarity_discriminator_table(gold_text, pred_text, should_match):
    gold = ClinicalRecord(conditions=[Condition(code_text=gold_text, clinical_status="active")])
    pred = ClinicalRecord(conditions=[Condition(code_text=pred_text, clinical_status="active")])
    s = score(pred, gold, pred_text)
    assert (s["tp"] == 1) is should_match

def test_unparsed_prediction_is_never_schema_valid():
    """A parse failure that fell back to an empty ClinicalRecord() must not be
    reported as schema-valid just because the empty record itself validates."""
    rec = ClinicalRecord()
    gold = ClinicalRecord()
    s = score(rec, gold, "note", parsed=False)
    assert s["schema_valid"] is False
