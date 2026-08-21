"""Tests for the tuned-vs-base model resolution in serve.py.

No live endpoint: OpenAIBackend/VLLMBackend construction is exercised
directly by llm_client's own tests, so here we mock `_run` (or
`build_backend`) rather than hit a network endpoint.
"""
import logging
from unittest import mock

from fastapi.testclient import TestClient

from fhir_extract import serve


def test_resolve_models_openai_returns_two_different_configured_ids():
    """Regression test: with backend: openai, tuned and base must resolve
    to the two distinct configured model ids, not the same one."""
    cfg = {"backend": "openai", "openai_tuned_model": "fhir-tuned",
           "openai_base_model": "Qwen/Qwen3-8B"}
    tuned, base = serve._resolve_models(cfg)
    assert tuned == "fhir-tuned"
    assert base == "Qwen/Qwen3-8B"
    assert tuned != base


def test_resolve_models_vllm_returns_tuned_and_base_model_paths():
    cfg = {"backend": "vllm", "tuned_model": "outputs/merged/qlora-8b",
           "base_model": "Qwen/Qwen3-8B"}
    tuned, base = serve._resolve_models(cfg)
    assert tuned == "outputs/merged/qlora-8b"
    assert base == "Qwen/Qwen3-8B"


def test_identical_models_logs_prominent_warning(caplog):
    cfg = {"backend": "openai", "openai_tuned_model": "same-id",
           "openai_base_model": "same-id"}
    with caplog.at_level(logging.WARNING, logger="fhir_extract.serve"):
        serve._check_identical_models("same-id", "same-id", cfg)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings
    message = warnings[0].getMessage()
    assert "openai_tuned_model" in message
    assert "openai_base_model" in message


def test_identical_models_vllm_warning_names_vllm_keys(caplog):
    cfg = {"backend": "vllm", "tuned_model": "same", "base_model": "same"}
    with caplog.at_level(logging.WARNING, logger="fhir_extract.serve"):
        serve._check_identical_models("same", "same", cfg)

    message = caplog.records[0].getMessage()
    assert "tuned_model" in message
    assert "base_model" in message


def test_distinct_models_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="fhir_extract.serve"):
        serve._check_identical_models("a", "b", {"backend": "vllm"})
    assert not caplog.records


def test_extract_response_includes_resolved_model_ids(monkeypatch):
    monkeypatch.setattr(serve, "_TUNED_MODEL", "fhir-tuned")
    monkeypatch.setattr(serve, "_BASE_MODEL", "Qwen/Qwen3-8B")

    def fake_run(model_id, note):
        return {"note_echo": note}, 1.0

    monkeypatch.setattr(serve, "_run", fake_run)

    client = TestClient(serve.app)
    resp = client.post("/extract", json={"note": "62yo M with HTN."})

    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == {"tuned": "fhir-tuned", "base": "Qwen/Qwen3-8B"}


def test_backend_for_openai_builds_separate_backend_per_model_id(monkeypatch):
    """Each resolved model id gets its own OpenAIBackend, both pointed at
    the shared base_url -- this is the fix for the original defect where a
    single shared backend served both tuned and base."""
    monkeypatch.setattr(serve, "cfg", {
        "backend": "openai", "base_url": "http://x/v1", "api_key": "EMPTY",
        "structured_output_mode": "guided_json",
    })
    serve._backends.clear()

    built = []

    def fake_build_backend(cfg):
        built.append(cfg)
        return mock.Mock()

    monkeypatch.setattr(serve, "build_backend", fake_build_backend)

    tuned_backend = serve._backend_for("fhir-tuned")
    base_backend = serve._backend_for("Qwen/Qwen3-8B")

    assert tuned_backend is not base_backend
    assert len(built) == 2
    assert {c["model"] for c in built} == {"fhir-tuned", "Qwen/Qwen3-8B"}
    assert all(c["base_url"] == "http://x/v1" for c in built)

    # cached on repeat call, no new backend built
    serve._backend_for("fhir-tuned")
    assert len(built) == 2


# --- demo fairness + path resolution -------------------------------------

def test_prompt_for_gives_the_base_model_the_schema():
    # The base model has never seen our field structure. Sending it the bare
    # training prompt makes it invent its own shape, so the side-by-side would
    # show the base failing for the wrong reason.
    from fhir_extract.serve import _prompt_for
    base = _prompt_for("qwen3-8b-base", "HR 72", tuned_id="fhir-lora")
    assert "loinc_code" in base


def test_prompt_for_gives_the_tuned_model_its_training_prompt():
    from fhir_extract.serve import _prompt_for
    from fhir_extract.baselines import EXTRACT_INSTRUCTION
    tuned = _prompt_for("fhir-lora", "HR 72", tuned_id="fhir-lora")
    assert tuned == EXTRACT_INSTRUCTION.format(note="HR 72")


def test_repo_paths_do_not_depend_on_the_working_directory():
    # serve.py resolved configs/serve.yaml and web/ against CWD at import, so
    # importing it from anywhere but the repo root raised (deferred M4).
    from fhir_extract.serve import _repo_path
    assert _repo_path("configs/serve.yaml").is_absolute()
    assert _repo_path("configs/serve.yaml").exists()
