"""Eval harness. Every number in the README comes from here."""
import json
import re
from pathlib import Path
from typing import Optional
import typer
import yaml

from .baselines import LLMBaseline, regex_extract
from .llm_client import build_backend
from .metrics import RESOURCES, aggregate, score
from .profile import ClinicalRecord

app = typer.Typer()


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def result_tag(system: str, model: str | None, shots: int, constrained: bool,
               schema_hint: bool, split: str) -> str:
    """Filename stem for a result JSON.

    Must distinguish every axis that changes the number, or runs silently
    overwrite each other: the fine-tuned and base models were both written
    to "llm_0shot_test_synthetic" and only the last one survived.
    """
    parts = [system]
    if system != 'regex' and model:
        parts.append(re.sub(r'[^A-Za-z0-9._-]', '-', model))
    parts.append(str(shots) + 'shot')
    if constrained:
        parts.append('constrained')
    if not schema_hint:
        parts.append('noschema')
    parts.append(split)
    return '_'.join(parts)


@app.command()
def main(
    config: str = "configs/data.yaml",
    split: str = "test_synthetic",
    system: str = "regex",
    model: Optional[str] = None,
    shots: int = 0,
    constrained: bool = False,
    schema_hint: bool = True,
    resources: str = "",
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

    # Restrict scoring for corpora that annotate only part of the profile
    # (ELMTEX has no vitals/allergies); default is every resource type.
    scored = tuple(r.strip() for r in resources.split(",") if r.strip()) or RESOURCES
    unknown = [r for r in scored if r not in RESOURCES]
    if unknown:
        raise ValueError(f"unknown resource type(s) {unknown}; expected from {list(RESOURCES)}")
    results = aggregate([score(p, g, n, parsed=parsed, resources=scored)
                         for (p, parsed), g, n in zip(preds, golds, notes)])
    results["system"] = system
    results["model"] = resolved_model if system != "regex" else None
    results["shots"] = shots
    results["constrained"] = constrained
    results["schema_hint"] = schema_hint
    results["scored_resources"] = list(scored)
    results["split"] = split

    out = Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    tag = result_tag(system, resolved_model, shots, constrained, schema_hint, split)
    (out / f"{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
