"""Generate clinical notes from label subsets. Resumable: re-running skips
encounters already present in the output file.
"""
import hashlib
import json
import random
from pathlib import Path
import typer
import yaml

from .prompts import build_prompt
from .subset import select_subset
from .synthea import iter_bundles, parse_bundle

app = typer.Typer()


def _done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["encounter_id"])
            except Exception:
                continue
    return done


def _encounter_rng(seed: int, encounter_id: str) -> random.Random:
    """Per-encounter RNG so a resumed run reproduces a fresh run exactly.

    Deriving the stream from (seed, encounter_id) rather than sharing one RNG
    across the work loop means a resumed run -- which skips already-done
    encounters before drawing -- produces the exact same subset and prompt
    variant per encounter as a fresh run. Order of processing and what was
    skipped no longer offset the stream.
    """
    digest = hashlib.sha256(f"{seed}:{encounter_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


@app.command()
def main(config: str = "configs/data.yaml") -> None:
    from vllm import LLM, SamplingParams

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out_path = Path(cfg["paths"]["interim"]) / "pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    typer.echo(f"Resuming: {len(done)} pairs already generated")

    # Build the work list deterministically.
    work = []
    for bundle in iter_bundles(Path(cfg["synthea"]["output_dir"])):
        for enc in parse_bundle(bundle):
            if enc.encounter_id in done:
                continue
            rng = _encounter_rng(cfg["seed"], enc.encounter_id)
            subset = select_subset(enc, rng)
            prompt, meta = build_prompt(enc, subset, rng)
            work.append((enc, subset, prompt, meta))
            if len(work) + len(done) >= cfg["generation"]["target_pairs"]:
                break
        if len(work) + len(done) >= cfg["generation"]["target_pairs"]:
            break

    typer.echo(f"Generating {len(work)} notes")
    llm = LLM(model=cfg["generation"]["model"], max_model_len=4096,
              gpu_memory_utilization=0.92)
    params = SamplingParams(
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        max_tokens=cfg["generation"]["max_tokens"],
    )

    batch = cfg["generation"]["batch_size"]
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(0, len(work), batch):
            chunk = work[i:i + batch]
            outputs = llm.generate([c[2] for c in chunk], params)
            for (enc, subset, _prompt, meta), out in zip(chunk, outputs):
                fh.write(json.dumps({
                    "patient_id": enc.patient_id,
                    "encounter_id": enc.encounter_id,
                    "note": out.outputs[0].text.strip(),
                    "label": subset.model_dump(),
                    "variant": meta,
                }) + "\n")
            fh.flush()   # checkpoint every batch
            typer.echo(f"  {min(i + batch, len(work))}/{len(work)}")


if __name__ == "__main__":
    app()
