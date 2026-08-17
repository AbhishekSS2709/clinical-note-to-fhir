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
    """Idempotent: a second call must not double-wrap."""
    template = "{% generation %}assistant body{% endgeneration %}"
    tok = _StubTokenizer(template)
    ensure_generation_markers(tok)
    assert tok.chat_template == template  # left untouched
    assert tok.chat_template.count("{% generation %}") == 1


def test_ensure_generation_markers_detects_whitespace_control_form():
    """The real patched template uses {%- generation -%}, not {% generation %}.

    A literal-substring early-exit would miss it and re-patch an already
    patched template. The implementation must gate on a regex.
    """
    template = "{%- generation -%}assistant body{%- endgeneration -%}"
    tok = _StubTokenizer(template)
    ensure_generation_markers(tok)
    assert tok.chat_template == template  # recognised as already patched


def test_ensure_generation_markers_does_not_gate_on_bare_word():
    """'generation' appears in every template via add_generation_prompt.

    Gating on that substring would make this a silent no-op and leave the
    assistant mask all zeros -- the exact bug this function exists to close.
    """
    template = "{%- if add_generation_prompt %}{{- 'x' }}{%- endif %}"
    tok = _StubTokenizer(template)
    with pytest.raises(ValueError):
        ensure_generation_markers(tok)


def test_ensure_generation_markers_refuses_unknown_template_shape():
    """Anchors must match exactly once; otherwise refuse rather than guess.

    Verified on the real Qwen3 template: the previous generic-scanner
    implementation returned OK while wrapping the <|im_start|>assistant
    header INSIDE the span -- a silent wrong span that trains the model on
    its own turn header. Refusing beats guessing.
    """
    tok = _StubTokenizer(_QWEN3_STYLE_TEMPLATE)
    with pytest.raises(ValueError, match="anchor not found"):
        ensure_generation_markers(tok)


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
