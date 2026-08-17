"""Tests for the LLM backend abstraction. No live endpoint, no vllm install,
and no module-scope import of vllm/openai -- everything here mocks
`openai.OpenAI` or stubs the `vllm` module.
"""
import sys
import time
import types
from unittest import mock

import pytest

from fhir_extract.llm_client import OpenAIBackend, VLLMBackend, build_backend


def _response(text: str):
    resp = mock.Mock()
    resp.choices = [mock.Mock(message=mock.Mock(content=text))]
    return resp


def test_build_backend_openai_returns_openai_backend():
    with mock.patch("openai.OpenAI"):
        backend = build_backend(
            {"backend": "openai", "base_url": "http://x/v1", "model": "m"})
    assert isinstance(backend, OpenAIBackend)


def test_build_backend_vllm_returns_vllm_backend(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = mock.Mock(return_value=mock.Mock())
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    backend = build_backend({"backend": "vllm", "model": "m"})
    assert isinstance(backend, VLLMBackend)


def test_build_backend_absent_backend_key_defaults_to_vllm(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = mock.Mock(return_value=mock.Mock())
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    backend = build_backend({"model": "m"})
    assert isinstance(backend, VLLMBackend)


def test_build_backend_unknown_raises_naming_options():
    with pytest.raises(ValueError, match="openai.*vllm|vllm.*openai"):
        build_backend({"backend": "carrier-pigeon"})


def test_openai_missing_model_raises_error_naming_config_key():
    with mock.patch("openai.OpenAI"):
        with pytest.raises(ValueError, match="model"):
            build_backend({"backend": "openai", "base_url": "http://x/v1"})


def test_complete_preserves_input_order_despite_out_of_order_completion():
    """The 10 mocked responses deliberately finish in reverse completion
    order (later-submitted requests sleep less and finish first). If
    `complete()` assembled results by completion order instead of indexing
    by input position, this would come back reversed.
    """
    def create(model, messages, **kwargs):
        index = int(messages[0]["content"].split("-")[1])
        time.sleep((9 - index) * 0.01)
        return _response(f"response-{index}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m", max_concurrency=10)
        prompts = [f"prompt-{i}" for i in range(10)]
        results = backend.complete(prompts, temperature=0.0, top_p=1.0,
                                    max_tokens=16, json_schema=None)

    assert results == [f"response-{i}" for i in range(10)]


def test_structured_output_mode_guided_json_uses_extra_body():
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="guided_json",
                                 enable_thinking=True)
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema={"type": "object"})

    assert captured["extra_body"] == {"guided_json": {"type": "object"}}
    assert "response_format" not in captured


def test_structured_output_mode_json_schema_uses_response_format():
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="json_schema",
                                 enable_thinking=True)
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema={"type": "object"})

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "clinical_record", "schema": {"type": "object"}},
    }
    assert "extra_body" not in captured


def test_structured_output_mode_none_sends_no_hint():
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="none",
                                 enable_thinking=True)
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema={"type": "object"})

    assert "extra_body" not in captured
    assert "response_format" not in captured


@pytest.mark.parametrize("mode", ["guided_json", "json_schema", "none"])
def test_no_json_schema_sends_no_hint_in_any_mode(mode):
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode=mode, enable_thinking=True)
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema=None)

    assert "extra_body" not in captured
    assert "response_format" not in captured


def test_enable_thinking_false_disables_thinking_via_chat_template_kwargs():
    """Default enable_thinking=False must suppress Qwen3's default-on
    thinking mode, which otherwise burns the token budget on reasoning
    tokens and can truncate the completion before any content is emitted."""
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="none")
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema=None)

    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_enable_thinking_true_sends_no_thinking_hint():
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="none", enable_thinking=True)
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema=None)

    assert "extra_body" not in captured


def test_enable_thinking_false_composes_with_guided_json_extra_body():
    """chat_template_kwargs and guided_json must both land in extra_body --
    neither should clobber the other."""
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="guided_json")
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema={"type": "object"})

    assert captured["extra_body"] == {
        "guided_json": {"type": "object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_enable_thinking_false_composes_with_json_schema_response_format():
    """structured_output_mode='json_schema' uses response_format, not
    extra_body, for the schema -- the thinking toggle must still land in
    extra_body alongside it without disturbing response_format."""
    captured = {}

    def create(model, messages, **kwargs):
        captured.update(kwargs)
        return _response("{}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m",
                                 structured_output_mode="json_schema")
        backend.complete(["p"], temperature=0.0, top_p=1.0, max_tokens=16,
                          json_schema={"type": "object"})

    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "clinical_record", "schema": {"type": "object"}},
    }


def test_single_failure_yields_empty_string_others_unaffected():
    def create(model, messages, **kwargs):
        prompt = messages[0]["content"]
        if prompt == "bad":
            raise RuntimeError("simulated 500 after retries")
        return _response(f"ok:{prompt}")

    with mock.patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = create
        backend = OpenAIBackend(base_url="http://x/v1", model="m")
        results = backend.complete(["good1", "bad", "good2"], temperature=0.0,
                                    top_p=1.0, max_tokens=16, json_schema=None)

    assert results == ["ok:good1", "", "ok:good2"]
