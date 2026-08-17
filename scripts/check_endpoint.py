"""Preflight check for an OpenAI-compatible inference endpoint.

Run this before a long generation/eval job to confirm the endpoint is up,
the configured model id is actually served, and structured output works the
way the pipeline expects.

Checks 2 and 3 route through `build_backend(...)` / `.complete()` -- the
exact same code path production uses (generate.py, baselines.py, serve.py).
A raw `openai.OpenAI` call here would skip the `enable_thinking` toggle and
give a false FAIL on a healthy Qwen3 endpoint (thinking mode burns the whole
token budget and returns empty content unless explicitly disabled).

Usage:
    python scripts/check_endpoint.py
    python scripts/check_endpoint.py --base-url http://host:8003/v1 --model Qwen/Qwen3-8B
    python scripts/check_endpoint.py --enable-thinking
"""
import argparse
import sys
from pathlib import Path

import yaml

from fhir_extract.llm_client import build_backend
from fhir_extract.profile import ClinicalRecord

_THINKING_HINT = (
    "  likely cause: Qwen3 is a hybrid thinking model with thinking mode "
    "ON by default; thinking burns the whole token budget and the "
    "completion comes back empty.\n"
    "  fix: set generation.enable_thinking: false in configs/data.yaml "
    "(or pass --enable-thinking=false)."
)


def _defaults() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["generation"]


def main() -> None:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=defaults.get("base_url"))
    parser.add_argument("--model", default=defaults.get("model"))
    parser.add_argument("--structured-output-mode",
                         default=defaults.get("structured_output_mode", "json_schema"),
                         choices=["guided_json", "json_schema", "none"])
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                         default=defaults.get("enable_thinking", False),
                         help="Qwen3 thinking mode; default from configs/data.yaml")
    args = parser.parse_args()

    backend = build_backend({
        "backend": "openai",
        "base_url": args.base_url,
        "model": args.model,
        "api_key": defaults.get("api_key", "EMPTY"),
        "timeout": defaults.get("timeout", 120),
        "max_retries": defaults.get("max_retries", 3),
        "structured_output_mode": args.structured_output_mode,
        "max_concurrency": 1,
        "enable_thinking": args.enable_thinking,
    })

    print(f"Checking endpoint {args.base_url}\n")
    results: dict[str, bool] = {}

    print("[1/3] GET /models")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=defaults.get("api_key", "EMPTY"))
        ids = [m.id for m in client.models.list().data]
        print(f"  found {len(ids)} model(s):")
        for mid in ids:
            print(f"    - {mid}")
        if args.model not in ids:
            print(f"  WARNING: configured model {args.model!r} not in the list above")
        results["models"] = True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  likely cause: endpoint unreachable or base_url is wrong.\n"
              "  fix: confirm the server is up and base_url points at it "
              "(e.g. http://host:port/v1).")
        results["models"] = False

    print(f"\n[2/3] trivial completion (enable_thinking={args.enable_thinking})")
    try:
        [text] = backend.complete(
            ["Say hello in one short sentence."],
            temperature=0.0, top_p=1.0, max_tokens=32, json_schema=None)
        print(f"  response: {text!r}")
        results["completion"] = bool(text)
        if not text:
            print(_THINKING_HINT)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results["completion"] = False

    print(f"\n[3/3] structured output (mode={args.structured_output_mode}, "
          f"enable_thinking={args.enable_thinking})")
    schema = ClinicalRecord.model_json_schema()
    try:
        [text] = backend.complete(
            ["Return an empty ClinicalRecord as JSON with keys "
             "conditions, medications, allergies, vitals, procedures, "
             "each an empty list."],
            temperature=0.0, top_p=1.0, max_tokens=256, json_schema=schema)
        print(f"  raw response: {text!r}")
        if not text:
            print(_THINKING_HINT)
            results["structured"] = False
        else:
            record = ClinicalRecord.model_validate_json(text)
            print(f"  parsed and validated OK: {record.model_dump()}")
            results["structured"] = True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  likely cause: the response didn't validate as JSON matching "
              "ClinicalRecord.\n"
              "  fix: confirm --structured-output-mode matches what the "
              "server supports (guided_json for vLLM guided decoding, "
              "json_schema for OpenAI-style response_format).")
        results["structured"] = False

    print("\n--- Summary ---")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
