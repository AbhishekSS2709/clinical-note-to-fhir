"""Tests for the pure-python compatibility shims in train.py.

None of these import torch/trl/transformers -- ensure_generation_markers
takes a lightweight tokenizer stub, and the dtype/warmup shims take stub
functions/dicts with the relevant signatures.
"""
import pytest

from fhir_extract.train import (
    ensure_generation_markers,
    resolve_dtype_kwarg,
    resolve_warmup_kwargs,
)


class _StubTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template


# A realistic Qwen3-style ChatML template, without {% generation %} markers.
_QWEN3_STYLE_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
)


def test_ensure_generation_markers_noop_when_already_present():
    template = "{% generation %}assistant body{% endgeneration %}"
    tok = _StubTokenizer(template)
    assert ensure_generation_markers(tok) is True
    assert tok.chat_template == template  # left untouched


def test_ensure_generation_markers_inserts_markers_into_qwen3_style_template():
    tok = _StubTokenizer(_QWEN3_STYLE_TEMPLATE)
    assert ensure_generation_markers(tok) is True
    assert "{% generation %}" in tok.chat_template
    assert "{% endgeneration %}" in tok.chat_template
    # The wrapped body still contains the assistant content expression.
    start = tok.chat_template.index("{% generation %}")
    end = tok.chat_template.index("{% endgeneration %}")
    assert "assistant" in tok.chat_template[start:end]


def test_ensure_generation_markers_raises_when_assistant_body_not_found():
    template = "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
    tok = _StubTokenizer(template)
    with pytest.raises(ValueError, match="assistant"):
        ensure_generation_markers(tok)


def test_ensure_generation_markers_raises_when_template_is_none():
    tok = _StubTokenizer(None)
    with pytest.raises(ValueError, match="chat_template"):
        ensure_generation_markers(tok)


# --- dtype shim -------------------------------------------------------

def test_resolve_dtype_kwarg_picks_dtype_when_supported():
    def stub_new(pretrained_model_name_or_path, *, dtype=None, **kw):
        pass

    assert resolve_dtype_kwarg(stub_new) == "dtype"


def test_resolve_dtype_kwarg_picks_torch_dtype_when_that_is_supported():
    def stub_old(pretrained_model_name_or_path, *, torch_dtype=None, **kw):
        pass

    assert resolve_dtype_kwarg(stub_old) == "torch_dtype"


# --- warmup shim --------------------------------------------------------

def test_resolve_warmup_kwargs_never_passes_both_keys():
    cfg = {"warmup_ratio": 0.03, "warmup_steps": 100, "num_train_epochs": 3}
    result = resolve_warmup_kwargs(cfg)
    assert not ({"warmup_ratio", "warmup_steps"} <= result.keys())
    assert result["warmup_ratio"] == 0.03
    assert result["num_train_epochs"] == 3


def test_resolve_warmup_kwargs_leaves_single_key_configs_untouched():
    cfg = {"warmup_ratio": 0.03}
    assert resolve_warmup_kwargs(cfg) == {"warmup_ratio": 0.03}


def test_resolve_warmup_kwargs_validates_against_target_signature():
    def stub_target(*, warmup_ratio=0.0, **kw):
        pass

    cfg = {"warmup_ratio": 0.03}
    assert resolve_warmup_kwargs(cfg, target=stub_target) == {"warmup_ratio": 0.03}
