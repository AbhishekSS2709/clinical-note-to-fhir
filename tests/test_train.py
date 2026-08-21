"""Tests for the pure-python compatibility shims in train.py.

None of these import torch/trl/transformers -- ensure_generation_markers
takes a lightweight tokenizer stub, and the dtype/warmup shims take stub
functions/dicts with the relevant signatures.
"""
import pytest

from fhir_extract.train import (
    ensure_generation_markers,
    resolve_dtype_kwarg,
    resolve_max_length_kwarg,
    resolve_warmup_kwargs,
    total_training_steps,
    unsupported_kwargs,
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


# --- total step count (drives warmup_ratio -> warmup_steps conversion) ----



def test_total_training_steps_multiplies_epochs_by_batches():
    cfg = {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
           "num_train_epochs": 3}
    # 17696 examples / (8*4) = 553 batches per epoch, x3 epochs
    assert total_training_steps(17696, cfg) == 553 * 3


def test_total_training_steps_rounds_a_partial_batch_up():
    cfg = {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
           "num_train_epochs": 1}
    assert total_training_steps(33, cfg) == 2


def test_total_training_steps_accounts_for_multiple_gpus():
    cfg = {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
           "num_train_epochs": 1}
    assert total_training_steps(6400, cfg, world_size=2) == 100


def test_warmup_ratio_converts_once_total_steps_is_supplied():
    # The real failure: SFTConfig on transformers 5.x rejects warmup_ratio, and
    # the call site passed no total_steps, so the run aborted at startup.
    # No **kwargs: _first_supported_kwarg treats a catch-all as accepting
    # anything, so a stub with **kw would not reject warmup_ratio at all.
    def target_without_ratio(*, warmup_steps=0, learning_rate=0.0):
        return None
    out = resolve_warmup_kwargs({"warmup_ratio": 0.03}, target=target_without_ratio,
                                total_steps=1659)
    assert out == {"warmup_steps": 50}


# --- SFTConfig key compatibility -----------------------------------------



def _sft_like(*, max_length=0, learning_rate=0.0, warmup_steps=0):
    """Stands in for trl 1.x SFTConfig: max_length, no max_seq_length."""
    return None


def test_max_seq_length_is_renamed_for_trl_1x():
    out = resolve_max_length_kwarg({"max_seq_length": 2048}, target=_sft_like)
    assert out == {"max_length": 2048}


def test_max_seq_length_is_left_alone_when_the_target_accepts_it():
    def old_sft(*, max_seq_length=0):
        return None
    assert resolve_max_length_kwarg({"max_seq_length": 2048}, target=old_sft) \
        == {"max_seq_length": 2048}


def test_unsupported_kwargs_reports_every_bad_key_at_once():
    # Discovering these one per launch costs a model load each time; the
    # pre-flight must list all of them in one raise.
    bad = unsupported_kwargs(
        {"learning_rate": 1e-4, "bogus_a": 1, "bogus_b": 2}, target=_sft_like)
    assert bad == ["bogus_a", "bogus_b"]


def test_unsupported_kwargs_is_empty_for_a_clean_config():
    assert unsupported_kwargs({"learning_rate": 1e-4}, target=_sft_like) == []
