from fhir_extract.profile import ClinicalRecord, Condition, MedicationStatement
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
