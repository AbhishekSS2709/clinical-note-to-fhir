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
from .profile import ClinicalRecord

cfg = yaml.safe_load(Path("configs/serve.yaml").read_text(encoding="utf-8"))
app = FastAPI(title="Clinical Note to FHIR")

_engines: dict[str, object] = {}


def _engine(path: str):
    if path not in _engines:
        from vllm import LLM
        _engines[path] = LLM(model=path, max_model_len=4096,
                             gpu_memory_utilization=cfg["gpu_memory_utilization"])
    return _engines[path]


class ExtractRequest(BaseModel):
    note: str


def _run(model_path: str, note: str) -> tuple[dict, float]:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    params = SamplingParams(
        temperature=0.0, max_tokens=1024,
        guided_decoding=GuidedDecodingParams(
            json=ClinicalRecord.model_json_schema()),
    )
    start = time.perf_counter()
    out = _engine(model_path).generate(
        [EXTRACT_INSTRUCTION.format(note=note)], params)
    elapsed = (time.perf_counter() - start) * 1000
    try:
        record = ClinicalRecord.model_validate_json(out[0].outputs[0].text.strip())
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
