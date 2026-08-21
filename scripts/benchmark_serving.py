"""Measure serving throughput and latency for the extraction models.

Answers the deployment question the F1 table cannot: what does this cost to
run? Reports tokens/sec, p50/p95 latency and prompt size at several
concurrency levels, then derives a cost per 1M notes from the measured rate.

The fine-tuned model's prompt carries no schema block and no exemplars, so the
interesting comparison is not just tokens/sec but tokens/sec *per note* --
few-shot spends ~8.7k prompt tokens per note where the fine-tuned model spends
~0.4k.

Usage:
    python scripts/benchmark_serving.py --base-url http://127.0.0.1:8004/v1 \
        --split data/processed/test_synthetic.jsonl --out outputs/serving_benchmark.json
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from fhir_extract.baselines import EXTRACT_INSTRUCTION, schema_instruction

app = typer.Typer()


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Explicit because statistics.quantiles
    interpolates, which misreports p95 on the small samples used here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def _run(client, model: str, prompts: list[str], concurrency: int,
         max_tokens: int) -> dict:
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0

    def one(prompt: str) -> tuple[float, int, int]:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (time.perf_counter() - start,
                resp.usage.prompt_tokens, resp.usage.completion_tokens)

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for latency, p_tok, c_tok in pool.map(one, prompts):
            latencies.append(latency)
            prompt_tokens += p_tok
            completion_tokens += c_tok
    wall = time.perf_counter() - wall_start

    return {
        "concurrency": concurrency,
        "notes": len(prompts),
        "wall_seconds": round(wall, 2),
        "notes_per_second": round(len(prompts) / wall, 3),
        "output_tokens_per_second": round(completion_tokens / wall, 1),
        "total_tokens_per_second": round((prompt_tokens + completion_tokens) / wall, 1),
        "prompt_tokens_per_note": round(prompt_tokens / len(prompts), 1),
        "output_tokens_per_note": round(completion_tokens / len(prompts), 1),
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
    }


@app.command()
def main(base_url: str = "http://127.0.0.1:8004/v1",
         split: str = "data/processed/test_synthetic.jsonl",
         out: str = "outputs/serving_benchmark.json",
         notes: int = 64,
         max_tokens: int = 1024,
         gpu_hour_usd: float = 0.80) -> None:
    """Benchmark base (5-shot and 0-shot) against the fine-tuned model.

    `gpu_hour_usd` defaults to a commodity A6000-class rental rate; the cost
    figure scales linearly, so substitute your own.
    """
    from openai import OpenAI

    rows = [json.loads(line) for line in
            Path(split).open(encoding="utf-8")][:notes]
    texts = [r["note"] for r in rows]
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)

    few_shot_examples = rows[:5]
    def few_shot_prompt(note: str) -> str:
        template = schema_instruction()
        prefix = ""
        for ex in few_shot_examples:
            prefix += template.format(note=ex["note"]) + json.dumps(ex["label"]) + "\n\n"
        return prefix + template.format(note=note)

    systems = {
        # The fine-tuned model is served the prompt it was trained on: no
        # schema block, no exemplars.
        "fine-tuned (0-shot)": ("fhir-lora", [EXTRACT_INSTRUCTION.format(note=n)
                                              for n in texts]),
        "base (0-shot + schema)": ("qwen3-8b-base",
                                   [schema_instruction().format(note=n) for n in texts]),
        "base (5-shot)": ("qwen3-8b-base", [few_shot_prompt(n) for n in texts]),
    }

    results = {}
    for name, (model, prompts) in systems.items():
        results[name] = []
        for concurrency in (1, 8, 32):
            typer.echo(f"{name} @ concurrency {concurrency} ...")
            measured = _run(client, model, prompts, concurrency, max_tokens)
            measured["usd_per_1m_notes"] = round(
                1_000_000 / measured["notes_per_second"] / 3600 * gpu_hour_usd, 2)
            results[name].append(measured)
            typer.echo(f"    {measured['notes_per_second']} notes/s, "
                       f"p95 {measured['latency_p95_s']}s, "
                       f"${measured['usd_per_1m_notes']}/1M notes")

    payload = {"gpu_hour_usd": gpu_hour_usd, "notes_sampled": len(texts),
               "max_tokens": max_tokens, "systems": results}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"wrote {out_path}")


if __name__ == "__main__":
    app()
