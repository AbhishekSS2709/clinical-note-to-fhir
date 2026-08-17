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

from .faithfulness import is_faithful
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
    for line in src.open(encoding="utf-8"):
        raw += 1
        row = json.loads(line)
        record = ClinicalRecord.model_validate(row["label"])
        if is_faithful(row["note"], record):
            kept.append(row)

    drop_rate = 1 - (len(kept) / raw) if raw else 0.0
    splits = split_by_patient(kept, cfg["splits"], cfg["seed"])

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
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "distinct_2": round(distinct_n([r["note"] for r in kept], 2), 4),
        "distinct_3": round(distinct_n([r["note"] for r in kept], 3), 4),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
