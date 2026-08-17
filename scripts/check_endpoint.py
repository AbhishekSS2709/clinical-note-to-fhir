"""Preflight check for an OpenAI-compatible inference endpoint.

Run this before a long generation/eval job to confirm the endpoint is up,
the configured model id is actually served, and structured output works the
way the pipeline expects.

Usage:
    python scripts/check_endpoint.py
    python scripts/check_endpoint.py --base-url http://host:8003/v1 --model Qwen/Qwen3-8B
"""
import argparse
import sys
from pathlib import Path

import yaml

from fhir_extract.profile import ClinicalRecord


def _defaults() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["generation"]


def main() -> None:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=defaults.get("base_url"))
    parser.add_argument("--model", default=defaults.get("model"))
    parser.add_argument("--structured-output-mode",
                         default=defaults.get("structured_output_mode", "json_schema"),
                         choices=["guided_json", "json_schema", "none"])
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=defaults.get("api_key", "EMPTY"))

    print(f"Checking endpoint {args.base_url}\n")
    results: dict[str, bool] = {}

    print("[1/3] GET /models")
    try:
        ids = [m.id for m in client.models.list().data]
        print(f"  found {len(ids)} model(s):")
        for mid in ids:
            print(f"    - {mid}")
        if args.model not in ids:
            print(f"  WARNING: configured model {args.model!r} not in the list above")
        results["models"] = True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results["models"] = False

    print("\n[2/3] trivial completion")
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=32,
        )
        text = resp.choices[0].message.content
        print(f"  response: {text!r}")
        results["completion"] = bool(text)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results["completion"] = False

    print(f"\n[3/3] structured output (mode={args.structured_output_mode})")
    schema = ClinicalRecord.model_json_schema()
    extra: dict = {}
    if args.structured_output_mode == "guided_json":
        extra = {"extra_body": {"guided_json": schema}}
    elif args.structured_output_mode == "json_schema":
        extra = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "clinical_record", "schema": schema}}}
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user",
                       "content": "Return an empty ClinicalRecord as JSON with keys "
                                  "conditions, medications, allergies, vitals, procedures, "
                                  "each an empty list."}],
            max_tokens=256,
            **extra,
        )
        text = resp.choices[0].message.content or ""
        print(f"  raw response: {text!r}")
        record = ClinicalRecord.model_validate_json(text)
        print(f"  parsed and validated OK: {record.model_dump()}")
        results["structured"] = True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        results["structured"] = False

    print("\n--- Summary ---")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
