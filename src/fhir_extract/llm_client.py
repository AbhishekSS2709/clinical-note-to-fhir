"""LLM backend abstraction: in-process vLLM or an OpenAI-compatible HTTP
endpoint.

Both backends implement `complete()`. Call sites (generate.py, baselines.py,
serve.py) build one via `build_backend(cfg)` and never touch vllm/openai
directly. Neither package is imported at module scope, so this module can be
imported with neither installed.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    def complete(self, prompts: list[str], *, temperature: float, top_p: float,
                 max_tokens: int, json_schema: dict | None) -> list[str]:
        """Return one completion string per prompt, in order."""


class VLLMBackend:
    """Wraps an in-process `vllm.LLM` engine."""

    def __init__(self, model: str, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.92):
        from vllm import LLM
        self.llm = LLM(model=model, max_model_len=max_model_len,
                        gpu_memory_utilization=gpu_memory_utilization)

    def complete(self, prompts: list[str], *, temperature: float, top_p: float,
                 max_tokens: int, json_schema: dict | None) -> list[str]:
        from vllm import SamplingParams
        guided = None
        if json_schema is not None:
            from vllm.sampling_params import GuidedDecodingParams
            guided = GuidedDecodingParams(json=json_schema)
        params = SamplingParams(temperature=temperature, top_p=top_p,
                                 max_tokens=max_tokens, guided_decoding=guided)
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text.strip() for out in outputs]


class OpenAIBackend:
    """Routes completions through an OpenAI-compatible HTTP endpoint (e.g. a
    vLLM OpenAI server). Structured output is configurable because servers
    differ in how they accept a JSON schema; see `structured_output_mode`.

    Requests are issued concurrently with a bounded worker pool. Results are
    indexed back to their input position, so they come back in prompt order
    regardless of completion order. A single request that fails after
    exhausting its retries yields "" for that prompt rather than aborting the
    batch.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 timeout: float = 120, max_retries: int = 3,
                 structured_output_mode: str = "guided_json",
                 max_concurrency: int = 8):
        if not model:
            raise ValueError(
                "config key 'model' is required for the openai backend (it "
                "must match a model id actually served at base_url). Run "
                "`curl <base_url>/models` to find it."
            )
        if structured_output_mode not in ("guided_json", "json_schema", "none"):
            raise ValueError(
                f"unknown structured_output_mode {structured_output_mode!r}; "
                "expected 'guided_json', 'json_schema', or 'none'"
            )
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout,
                              max_retries=max_retries)
        self.model = model
        self.structured_output_mode = structured_output_mode
        self.max_concurrency = max_concurrency

    def _structured_kwargs(self, json_schema: dict | None) -> dict:
        if json_schema is None or self.structured_output_mode == "none":
            return {}
        if self.structured_output_mode == "guided_json":
            return {"extra_body": {"guided_json": json_schema}}
        return {"response_format": {"type": "json_schema", "json_schema": {
            "name": "clinical_record", "schema": json_schema}}}

    def complete(self, prompts: list[str], *, temperature: float, top_p: float,
                 max_tokens: int, json_schema: dict | None) -> list[str]:
        results: list[str] = [""] * len(prompts)
        extra = self._structured_kwargs(json_schema)

        def _one(index: int, prompt: str) -> tuple[int, str]:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    **extra,
                )
                return index, (resp.choices[0].message.content or "").strip()
            except Exception:
                logger.warning("completion %d failed after retries; returning "
                                "empty string", index, exc_info=True)
                return index, ""

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [pool.submit(_one, i, p) for i, p in enumerate(prompts)]
            for future in as_completed(futures):
                index, text = future.result()
                results[index] = text
        return results


def build_backend(cfg: dict) -> LLMBackend:
    """Choose a backend by cfg['backend'] ('openai' or 'vllm', default 'vllm')."""
    backend = cfg.get("backend", "vllm")
    if backend == "openai":
        return OpenAIBackend(
            base_url=cfg["base_url"],
            model=cfg.get("model"),
            api_key=cfg.get("api_key", "EMPTY"),
            timeout=cfg.get("timeout", 120),
            max_retries=cfg.get("max_retries", 3),
            structured_output_mode=cfg.get("structured_output_mode", "guided_json"),
            max_concurrency=cfg.get("max_concurrency", 8),
        )
    if backend == "vllm":
        return VLLMBackend(
            model=cfg["model"],
            max_model_len=cfg.get("max_model_len", 4096),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.92),
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'openai' or 'vllm'")
