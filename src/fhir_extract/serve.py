"""Serve tuned and base models side by side.

The side-by-side response is the point: it renders model quality visible in
three seconds to someone who will never read the ablation table.
See spec section 10.
"""
import logging
import time
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .baselines import EXTRACT_INSTRUCTION, parse_record
from .llm_client import build_backend
from .profile import ClinicalRecord

logger = logging.getLogger(__name__)

cfg = yaml.safe_load(Path("configs/serve.yaml").read_text(encoding="utf-8"))
app = FastAPI(title="Clinical Note to FHIR")

_backends: dict[str, object] = {}


def _resolve_models(cfg: dict) -> tuple[str, str]:
    """Return (tuned_id, base_id) per cfg['backend'].

    backend: openai -- two model ids served by one shared endpoint
    (`openai_tuned_model` / `openai_base_model`). backend: vllm -- two
    local model paths, each its own in-process engine (`tuned_model` /
    `base_model`). See docs/decisions/openai-backend.md.
    """
    if cfg.get("backend", "vllm") == "openai":
        return cfg["openai_tuned_model"], cfg["openai_base_model"]
    return cfg["tuned_model"], cfg["base_model"]


def _check_identical_models(tuned_id: str, base_id: str, cfg: dict) -> None:
    """Warn loudly if tuned and base resolve to the same model id -- the
    side-by-side comparison is then meaningless (identical outputs). Does
    not raise: smoke-testing the plumbing before a fine-tune exists is a
    legitimate use case.
    """
    if tuned_id != base_id:
        return
    keys = ("'openai_tuned_model' / 'openai_base_model'"
            if cfg.get("backend", "vllm") == "openai"
            else "'tuned_model' / 'base_model'")
    logger.warning(
        "!!! TUNED AND BASE MODEL BOTH RESOLVE TO %r -- the side-by-side "
        "comparison is MEANINGLESS, both panes will show identical output. "
        "Check config keys %s in configs/serve.yaml. !!!",
        tuned_id, keys,
    )


_TUNED_MODEL, _BASE_MODEL = _resolve_models(cfg)
_check_identical_models(_TUNED_MODEL, _BASE_MODEL, cfg)


def _backend_for(model_id: str):
    """One backend per resolved model id, cached.

    backend: vllm -- model_id is a local model path; each gets its own
    in-process engine. backend: openai -- model_id is a model id served by
    the shared `base_url` endpoint (e.g. the base model and a LoRA adapter
    exposed via `vllm serve ... --lora-modules`); each gets its own
    OpenAIBackend pointed at that same endpoint. See
    docs/decisions/openai-backend.md.
    """
    if model_id not in _backends:
        if cfg.get("backend", "vllm") == "openai":
            _backends[model_id] = build_backend({
                "backend": "openai",
                "base_url": cfg["base_url"],
                "model": model_id,
                "api_key": cfg.get("api_key", "EMPTY"),
                "structured_output_mode": cfg.get("structured_output_mode", "json_schema"),
                "max_concurrency": cfg.get("max_concurrency", 8),
                "timeout": cfg.get("timeout", 120),
                "max_retries": cfg.get("max_retries", 3),
                "enable_thinking": cfg.get("enable_thinking", False),
            })
        else:
            _backends[model_id] = build_backend({
                "backend": "vllm", "model": model_id,
                "gpu_memory_utilization": cfg["gpu_memory_utilization"],
            })
    return _backends[model_id]


class ExtractRequest(BaseModel):
    note: str


def _run(model_id: str, note: str) -> tuple[dict, float]:
    start = time.perf_counter()
    texts = _backend_for(model_id).complete(
        [EXTRACT_INSTRUCTION.format(note=note)],
        temperature=0.0, top_p=1.0, max_tokens=1024,
        json_schema=ClinicalRecord.model_json_schema(),
    )
    elapsed = (time.perf_counter() - start) * 1000
    record, _parsed = parse_record(texts[0])
    return record.model_dump(), round(elapsed, 1)


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    tuned, tuned_ms = _run(_TUNED_MODEL, req.note)
    base, base_ms = _run(_BASE_MODEL, req.note)
    return {"tuned": tuned, "base": base,
            "latency_ms": {"tuned": tuned_ms, "base": base_ms},
            "models": {"tuned": _TUNED_MODEL, "base": _BASE_MODEL}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
