"""Filter generated pairs and split them by patient.

Splitting by patient rather than by note prevents patient-specific phrasing
leaking across the train/test boundary. See spec section 4.6.
"""
import json
import random
from collections import defaultdict
from pathlib import Path
import typer
import yaml

from .faithfulness import unanchored_facts, invented_facts
from .profile import ClinicalRecord

app = typer.Typer()


def distinct_n(texts: list[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams. Low values mean repetition."""
    total, unique = 0, set()
    for t in texts:
        tokens = t.lower().split()
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def reserve_allergy_slice(rows: list[dict], target_rows: int,
                          seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out a dedicated AllergyIntolerance evaluation slice.

    Synthea records one allergy per patient, so allergy-labelled rows are ~2%
    of the corpus. A random patient split puts roughly a dozen of them in a
    500-row test set -- too few to report an F1 against. This reserves whole
    patients (never individual rows, which would leak a patient's phrasing
    across the boundary) until the slice holds `target_rows` allergy-labelled
    rows, and leaves the rest in the pool so train still sees allergies.

    The slice is deliberately NOT part of test_synthetic: it is enriched, so
    folding it in would skew the headline distribution.
    """
    if target_rows <= 0:
        return [], list(rows)

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    eligible = sorted(p for p, rs in by_patient.items()
                      if any(r["label"].get("allergies") for r in rs))
    random.Random(seed).shuffle(eligible)

    held: set[str] = set()
    count = 0
    for patient in eligible:
        if count >= target_rows:
            break
        held.add(patient)
        count += sum(1 for r in by_patient[patient] if r["label"].get("allergies"))

    sliced = [r for r in rows if r["patient_id"] in held]
    rest = [r for r in rows if r["patient_id"] not in held]
    return sliced, rest


def split_by_patient(rows: list[dict], ratios: dict[str, int],
                     seed: int) -> dict[str, list[dict]]:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    splits: dict[str, list[dict]] = {k: [] for k in ratios}
    # Fill held-out splits to their targets first; everything else overflows to train.
    order = [k for k in ratios if k != "train"]
    idx = 0
    for patient in patients:
        while idx < len(order) and len(splits[order[idx]]) >= ratios[order[idx]]:
            idx += 1
        target = order[idx] if idx < len(order) else "train"
        splits[target].extend(by_patient[patient])
    return splits


@app.command()
def main(config: str = "configs/data.yaml") -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    src = Path(cfg["paths"]["interim"]) / "pairs.jsonl"
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, kept = 0, []
    missing_count, invented_count = 0, 0
    for line in src.open(encoding="utf-8"):
        raw += 1
        row = json.loads(line)
        record = ClinicalRecord.model_validate(row["label"])
        missing = unanchored_facts(row["note"], record)
        invented = invented_facts(row["note"], record)
        if missing:
            missing_count += 1
        if invented:
            invented_count += 1
        if not missing and not invented:
            kept.append(row)

    rows_by_resource = {
        k: sum(1 for r in kept if r["label"].get(k))
        for k in ("conditions", "medications", "allergies", "vitals", "procedures")
    }
    drop_rate = 1 - (len(kept) / raw) if raw else 0.0
    drop_rate_missing = missing_count / raw if raw else 0.0
    drop_rate_invented = invented_count / raw if raw else 0.0
    # Pull the enriched allergy slice out BEFORE the main split, so
    # test_synthetic stays distribution-faithful and no patient spans both.
    split_targets = dict(cfg["splits"])
    allergy_target = split_targets.pop("test_allergies", 0)
    # `kept` must stay the full filtered corpus -- kept_pairs and distinct_n
    # below are reported over it, slice included.
    allergy_slice, pool = reserve_allergy_slice(kept, allergy_target, cfg["seed"])
    splits = split_by_patient(pool, split_targets, cfg["seed"])
    if allergy_target:
        splits["test_allergies"] = allergy_slice

    for name, rows in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    manifest = {
        "seed": cfg["seed"],
        "generation_model": cfg["generation"]["model"],
        "raw_pairs": raw,
        "kept_pairs": len(kept),
        "drop_rate": round(drop_rate, 4),
        "drop_rate_missing": round(drop_rate_missing, 4),
        "drop_rate_invented": round(drop_rate_invented, 4),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "distinct_2": round(distinct_n([r["note"] for r in kept], 2), 4),
        "distinct_3": round(distinct_n([r["note"] for r in kept], 3), 4),
        # Corpus composition. Recorded because the corpus is now built from
        # several generation passes with different encounter_filter settings,
        # and the per-resource results are only as meaningful as these counts.
        "rows_by_resource": rows_by_resource,
        "allergy_rows_in_slice": sum(
            1 for r in allergy_slice if r["label"].get("allergies")),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
