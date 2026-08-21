"""Render the README results tables from outputs/eval/*.json.

The README claims no number in it is typed by hand. This is what makes that
true: run it, paste the output, and every cell traces to a result file. It also
serves as a check -- if a table drifts from the JSON, the diff shows up here.

Usage:
    python scripts/results_table.py                 # all splits
    python scripts/results_table.py --split elmtex_test
"""
import json
from pathlib import Path

import typer

app = typer.Typer()

# Display order and labels. Anything not listed still prints, after these.
ORDER = ["regex", "base-0", "base-0-constr", "base-5", "ft-bf16", "ft-qlora", "ft-elmtex"]
def label_for(key: str) -> str:
    """Human label, including per-checkpoint labels for the v2 curve."""
    if key.startswith("ft-elmtex-"):
        return f"Qwen3-8B + LoRA on real reports (v2, step {key.rsplit('-', 1)[-1]})"
    return LABELS.get(key, key)


LABELS = {
    "regex": "Regex baseline",
    "base-0": "Qwen3-8B 0-shot (schema in prompt)",
    "base-0-constr": "Qwen3-8B 0-shot + constrained decoding",
    "base-5": "Qwen3-8B 5-shot",
    "ft-bf16": "**Qwen3-8B + LoRA bf16 (this project)**",
    "ft-qlora": "Qwen3-8B + QLoRA 4-bit (ablation)",
    "ft-elmtex": "**Qwen3-8B + LoRA on real reports (v2)**",
}


def classify(result: dict) -> str:
    """Map a result file onto a stable row key."""
    if result["system"] == "regex":
        return "regex"
    model = (result.get("model") or "").lower()
    if "qlora" in model:
        return "ft-qlora"
    # "fhir-v2" is the served name for the ELMTEX-trained adapter; without it
    # this fell through to the base-model row and overwrote it. The trailing
    # checkpoint number must survive too, or the whole data-efficiency curve
    # collapses onto a single row.
    if "elmtex" in model or "v2" in model:
        step = model.rsplit("-", 1)[-1]
        return f"ft-elmtex-{step}" if step.isdigit() else "ft-elmtex"
    if "fhir-lora" in model:
        return "ft-bf16"
    key = f"base-{result['shots']}"
    return key + "-constr" if result.get("constrained") else key


def load(eval_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(eval_dir.glob("*.json"))]


@app.command()
def main(eval_dir: str = "outputs/eval", split: str = "") -> None:
    results = load(Path(eval_dir))
    splits = [split] if split else sorted({r["split"] for r in results})

    for name in splits:
        rows = [r for r in results if r["split"] == name]
        if not rows:
            continue
        scored = rows[0].get("scored_resources") or []
        header = f"### {name} (n={rows[0]['n']}"
        header += f", scored: {', '.join(scored)})" if scored else ")"
        typer.echo(f"\n{header}\n")
        typer.echo("| System | Micro-F1 | Macro-F1 | Schema validity | "
                   "Hallucinated facts/note | Omitted facts/note |")
        typer.echo("|---|---:|---:|---:|---:|---:|")

        by_key = {classify(r): r for r in rows}
        keys = [k for k in ORDER if k in by_key]
        keys += [k for k in by_key if k not in ORDER]
        for key in keys:
            r = by_key[key]
            typer.echo(f"| {label_for(key)} | {r['micro_f1']:.3f} | "
                       f"{r['macro_f1']:.3f} | {r['schema_validity']:.3f} | "
                       f"{r['hallucination_rate']:.3f} | {r['omission_rate']:.2f} |")

        typer.echo("\nPer-resource F1:\n")
        types = list(rows[0]["per_resource"])
        typer.echo("| System | " + " | ".join(types) + " |")
        typer.echo("|---" * (len(types) + 1) + "|")
        for key in keys:
            pr = by_key[key]["per_resource"]
            typer.echo(f"| {label_for(key)} | "
                       + " | ".join(f"{pr[t]:.3f}" for t in types) + " |")


if __name__ == "__main__":
    app()
