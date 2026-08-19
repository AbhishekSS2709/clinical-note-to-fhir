"""Eval harness. Every number in the README comes from here."""
import json
from pathlib import Path
from typing import Optional
import typer
import yaml

from .baselines import LLMBaseline, regex_extract
from .llm_client import build_backend
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
    model: Optional[str] = None,
    shots: int = 0,
    constrained: bool = False,
    schema_hint: bool = True,
) -> None:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    split_path = Path(cfg["paths"]["processed"]) / f"{split}.jsonl"
    rows = _load(split_path)
    if not rows:
        raise ValueError(f"split '{split}' is empty: {split_path} contains no rows; "
                         "refusing to write a meaningless 0.0 result")
    notes = [r["note"] for r in rows]
    golds = [ClinicalRecord.model_validate(r["label"]) for r in rows]

    resolved_model = model

    if system == "regex":
        preds = [(regex_extract(n), True) for n in notes]
    else:
        if "inference" not in cfg:
            raise ValueError(
                f"config {config!r} has no 'inference:' block, required for "
                f"system={system!r}; add one (see configs/data.yaml) rather "
                "than relying on a hardcoded default backend."
            )
        inference_cfg = dict(cfg["inference"])
        if model is not None:
            inference_cfg["model"] = model
        resolved_model = inference_cfg.get("model")
        backend = build_backend(inference_cfg)

        # Load examples only if shots > 0
        if shots > 0:
            train_path = Path(cfg["paths"]["processed"]) / "train.jsonl"
            if not train_path.exists():
                raise FileNotFoundError(
                    f"few-shot evaluation requires train.jsonl, but not found at: {train_path}. "
                    f"Zero-shot evaluation (--shots 0) does not need a train split."
                )
            examples = _load(train_path)[:shots]
        else:
            examples = []

        # schema_hint=False for the fine-tuned model: it must be evaluated on
        # the prompt it was trained on, which carries no schema block.
        preds = LLMBaseline(resolved_model, shots, constrained, examples,
                             backend=backend,
                             inference_config=inference_cfg,
                             schema_hint=schema_hint).extract_batch(notes)

    results = aggregate([score(p, g, n, parsed=parsed)
                         for (p, parsed), g, n in zip(preds, golds, notes)])
    results["system"] = system
    results["model"] = resolved_model if system != "regex" else None
    results["shots"] = shots
    results["constrained"] = constrained
    results["schema_hint"] = schema_hint
    results["split"] = split

    out = Path("outputs/eval"); out.mkdir(parents=True, exist_ok=True)
    tag = f"{system}_{shots}shot{'_constrained' if constrained else ''}_{split}"
    (out / f"{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
