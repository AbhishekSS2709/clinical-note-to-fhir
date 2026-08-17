"""Eval harness. Every number in the README comes from here."""
import json
from pathlib import Path
import typer
import yaml

from .baselines import LLMBaseline, regex_extract
from .metrics import aggregate, score
from .profile import ClinicalRecord

app = typer.Typer()


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


@app.command()
def main(
    config: str = "configs/data.yaml",
    split: str = "test_synthetic",
    system: str = "regex",
    model: str = "Qwen/Qwen3-8B",
    shots: int = 0,
    constrained: bool = False,
) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    rows = _load(Path(cfg["paths"]["processed"]) / f"{split}.jsonl")
    notes = [r["note"] for r in rows]
    golds = [ClinicalRecord.model_validate(r["label"]) for r in rows]

    if system == "regex":
        preds = [regex_extract(n) for n in notes]
    else:
        examples = _load(Path(cfg["paths"]["processed"]) / "train.jsonl")[:shots]
        preds = LLMBaseline(model, shots, constrained, examples).extract_batch(notes)

    results = aggregate([score(p, g, n) for p, g, n in zip(preds, golds, notes)])
    results["system"] = system
    results["model"] = model if system != "regex" else None
    results["shots"] = shots
    results["constrained"] = constrained
    results["split"] = split

    out = Path("outputs/eval"); out.mkdir(parents=True, exist_ok=True)
    tag = f"{system}_{shots}shot{'_constrained' if constrained else ''}_{split}"
    (out / f"{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
