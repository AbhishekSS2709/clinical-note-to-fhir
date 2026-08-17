"""Serve tuned and base models side by side.

The side-by-side response is the point: it renders model quality visible in
three seconds to someone who will never read the ablation table.
See spec section 10.
"""
import time
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .baselines import EXTRACT_INSTRUCTION
from .llm_client import build_backend
from .profile import ClinicalRecord

cfg = yaml.safe_load(Path("configs/serve.yaml").read_text(encoding="utf-8"))
app = FastAPI(title="Clinical Note to FHIR")

_backends: dict[str, object] = {}


def _backend_for(model_path: str):
    """One backend per distinct local model path (vllm), or a single shared
    backend for the openai HTTP endpoint (both tuned and base then route
    through the same served model -- see docs/decisions/openai-backend.md).
    """
    key = "openai" if cfg.get("backend", "vllm") == "openai" else model_path
    if key not in _backends:
        if key == "openai":
            _backends[key] = build_backend(cfg)
        else:
            _backends[key] = build_backend({
                "backend": "vllm", "model": model_path,
                "gpu_memory_utilization": cfg["gpu_memory_utilization"],
            })
    return _backends[key]


class ExtractRequest(BaseModel):
    note: str


def _run(model_path: str, note: str) -> tuple[dict, float]:
    start = time.perf_counter()
    texts = _backend_for(model_path).complete(
        [EXTRACT_INSTRUCTION.format(note=note)],
        temperature=0.0, top_p=1.0, max_tokens=1024,
        json_schema=ClinicalRecord.model_json_schema(),
    )
    elapsed = (time.perf_counter() - start) * 1000
    try:
        record = ClinicalRecord.model_validate_json(texts[0])
    except Exception:
        record = ClinicalRecord()
    return record.model_dump(), round(elapsed, 1)


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    tuned, tuned_ms = _run(cfg["tuned_model"], req.note)
    base, base_ms = _run(cfg["base_model"], req.note)
    return {"tuned": tuned, "base": base,
            "latency_ms": {"tuned": tuned_ms, "base": base_ms}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
