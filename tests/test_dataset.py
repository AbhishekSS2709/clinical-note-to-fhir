from fhir_extract.dataset import split_by_patient, distinct_n
from fhir_extract.dataset import reserve_allergy_slice

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




def _allergy_rows():
    # 6 patients x 3 rows; patients p0-p3 have one allergy-labelled row each.
    rows = []
    for p in range(6):
        for i in range(3):
            label = {"allergies": [{"substance_text": "penicillin"}]} \
                if (p < 4 and i == 0) else {"allergies": []}
            rows.append({"patient_id": f"p{p}", "note": f"n{p}-{i}", "label": label})
    return rows


def test_slice_takes_whole_patients_so_none_leaks_into_the_remainder():
    sl, rest = reserve_allergy_slice(_allergy_rows(), target_rows=2, seed=1)
    assert {r["patient_id"] for r in sl} & {r["patient_id"] for r in rest} == set()


def test_slice_reaches_the_target_count_of_allergy_rows():
    sl, _ = reserve_allergy_slice(_allergy_rows(), target_rows=3, seed=1)
    assert sum(1 for r in sl if r["label"].get("allergies")) >= 3


def test_slice_is_capped_by_what_exists_rather_than_raising():
    sl, rest = reserve_allergy_slice(_allergy_rows(), target_rows=99, seed=1)
    assert sum(1 for r in sl if r["label"].get("allergies")) == 4
    assert {r["patient_id"] for r in rest} == {"p4", "p5"}


def test_slice_stops_early_instead_of_consuming_every_allergy_patient():
    sl, rest = reserve_allergy_slice(_allergy_rows(), target_rows=1, seed=1)
    assert len({r["patient_id"] for r in sl}) == 1
    # "any allergy row remains" is too weak: on the real corpus a target of
    # 150 against 165 available rows passed that check while leaving 15 rows
    # to train on. Require a real majority to survive for training.
    remaining = sum(1 for r in rest if r["label"].get("allergies"))
    assert remaining >= 3, f"only {remaining} allergy rows left for training"


def test_zero_target_reserves_nothing():
    sl, rest = reserve_allergy_slice(_allergy_rows(), target_rows=0, seed=1)
    assert sl == [] and len(rest) == 18


def test_slice_is_deterministic_for_a_seed():
    a, _ = reserve_allergy_slice(_allergy_rows(), target_rows=2, seed=7)
    b, _ = reserve_allergy_slice(_allergy_rows(), target_rows=2, seed=7)
    assert [r["note"] for r in a] == [r["note"] for r in b]
