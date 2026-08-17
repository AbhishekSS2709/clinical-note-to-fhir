from fhir_extract.dataset import split_by_patient, distinct_n

def _rows():
    return [{"patient_id": f"p{i // 5}", "note": f"note {i}", "label": {}}
            for i in range(50)]

def test_no_patient_appears_in_two_splits():
    splits = split_by_patient(_rows(), {"train": 30, "val": 10,
                                        "test_synthetic": 10}, seed=1)
    seen = {}
    for name, rows in splits.items():
        for r in rows:
            assert seen.setdefault(r["patient_id"], name) == name, \
                "patient leaked across splits"

def test_splits_are_deterministic_for_a_seed():
    a = split_by_patient(_rows(), {"train": 30, "val": 10, "test_synthetic": 10}, seed=1)
    b = split_by_patient(_rows(), {"train": 30, "val": 10, "test_synthetic": 10}, seed=1)
    assert [r["note"] for r in a["train"]] == [r["note"] for r in b["train"]]

def test_all_rows_are_assigned():
    splits = split_by_patient(_rows(), {"train": 30, "val": 10,
                                        "test_synthetic": 10}, seed=1)
    assert sum(len(v) for v in splits.values()) == 50

def test_distinct_n_detects_repetition():
    varied = ["the cat sat on the mat", "a dog ran through the park"]
    same = ["the cat sat on the mat"] * 2
    assert distinct_n(varied, 2) > distinct_n(same, 2)

def test_held_out_splits_are_filled_before_train():
    rows = [{"patient_id": f"p{i//4}", "note": "n", "label": {}} for i in range(8000)]
    s = split_by_patient(rows, {"train": 8000, "val": 500, "test_synthetic": 500}, seed=42)
    assert s["val"] >= [] and len(s["val"]) > 0 and len(s["test_synthetic"]) > 0
    assert sum(len(v) for v in s.values()) == 8000

def test_no_patient_leaks_after_fill_order_change():
    rows = [{"patient_id": f"p{i//4}", "note": "n", "label": {}} for i in range(400)]
    s = split_by_patient(rows, {"train": 300, "val": 50, "test_synthetic": 50}, seed=1)
    seen = {}
    for name, rs in s.items():
        for r in rs:
            assert seen.setdefault(r["patient_id"], name) == name
