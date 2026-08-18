"""Generate clinical notes from label subsets. Resumable: re-running skips
encounters already present in the output file.
"""
import hashlib
import json
import random
from pathlib import Path
import typer
import yaml

from .llm_client import build_backend
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
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out_path = Path(cfg["paths"]["interim"]) / "pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    typer.echo(f"Resuming: {len(done)} pairs already generated")

    # Build the work list deterministically.
    # Only ~33% of Synthea encounters carry vitals. Generating notes for the
    # rest wastes endpoint time and under-represents the numeric extraction
    # (BP/HR) the task cares about, so skip them unless asked not to.
    require_vitals = cfg["generation"].get("require_vitals", True)
    skipped_no_vitals = 0

    work = []
    for bundle in iter_bundles(Path(cfg["synthea"]["output_dir"])):
        for enc in parse_bundle(bundle):
            if enc.encounter_id in done:
                continue
            if require_vitals and not enc.record.vitals:
                skipped_no_vitals += 1
                continue
            rng = _encounter_rng(cfg["seed"], enc.encounter_id)
            subset = select_subset(enc, rng)
            # A subset emptied by the narratability filter has nothing to
            # write a note about; labelling it would teach the model to
            # return {} on notes that do contain findings.
            if not any([subset.conditions, subset.medications, subset.allergies,
                        subset.vitals, subset.procedures]):
                continue
            prompt, meta = build_prompt(enc, subset, rng)
            work.append((enc, subset, prompt, meta))
            if len(work) + len(done) >= cfg["generation"]["target_pairs"]:
                break
        if len(work) + len(done) >= cfg["generation"]["target_pairs"]:
            break

    typer.echo(f"Generating {len(work)} notes "
               f"(skipped {skipped_no_vitals} encounters with no vitals)")
    backend = build_backend(cfg["generation"])

    batch = cfg["generation"]["batch_size"]
    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(0, len(work), batch):
            chunk = work[i:i + batch]
            texts = backend.complete(
                [c[2] for c in chunk],
                temperature=cfg["generation"]["temperature"],
                top_p=cfg["generation"]["top_p"],
                max_tokens=cfg["generation"]["max_tokens"],
                json_schema=None,
            )
            for (enc, subset, _prompt, meta), text in zip(chunk, texts):
                fh.write(json.dumps({
                    "patient_id": enc.patient_id,
                    "encounter_id": enc.encounter_id,
                    "note": text,
                    "label": subset.model_dump(),
                    "variant": meta,
                }) + "\n")
            fh.flush()   # checkpoint every batch
            typer.echo(f"  {min(i + batch, len(work))}/{len(work)}")


if __name__ == "__main__":
    app()
