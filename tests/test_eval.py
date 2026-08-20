"""Runnable proof that Ruling J's `constrained` plumbing survives end to end,
without importing vllm (system="regex" never touches LLMBaseline).
"""
import json
from pathlib import Path

import pytest
import yaml

from fhir_extract.eval import main, result_tag


def _write_fixture(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    row = {"patient_id": "p1", "note": "HR 78 bpm", "label": {}}
    (processed / "test_synthetic.jsonl").write_text(json.dumps(row) + "\n",
                                                      encoding="utf-8")
    (processed / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    config = tmp_path / "data.yaml"
    config.write_text(yaml.dump({"paths": {"processed": str(processed)}}),
                       encoding="utf-8")
    return config


def test_constrained_flag_changes_output_tag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _write_fixture(tmp_path)

    main(config=str(config), system="regex", constrained=False)
    main(config=str(config), system="regex", constrained=True)

    unconstrained = tmp_path / "outputs/eval/regex_0shot_test_synthetic.json"
    constrained = tmp_path / "outputs/eval/regex_0shot_constrained_test_synthetic.json"
    assert unconstrained.exists()
    assert constrained.exists()

    assert json.loads(unconstrained.read_text())["constrained"] is False
    assert json.loads(constrained.read_text())["constrained"] is True


def _write_llm_fixture(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    row = {"patient_id": "p1", "note": "HR 78 bpm", "label": {}}
    (processed / "test_synthetic.jsonl").write_text(json.dumps(row) + "\n",
                                                      encoding="utf-8")
    (processed / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    config = tmp_path / "data.yaml"
    config.write_text(yaml.dump({
        "paths": {"processed": str(processed)},
        "inference": {
            "backend": "openai",
            "model": "Qwen/Qwen3-8B",
            "base_url": "http://10.24.6.107:8003/v1",
            "api_key": "EMPTY",
        },
    }), encoding="utf-8")
    return config


class _FakeBackend:
    def complete(self, prompts, *, temperature, top_p, max_tokens, json_schema):
        return ['{"conditions": [], "medications": [], "allergies": [], '
                '"vitals": [], "procedures": []}' for _ in prompts]


def test_eval_llm_system_builds_backend_from_inference_config(tmp_path, monkeypatch):
    """eval.py must build the backend from cfg['inference'] via build_backend
    -- no more hardcoded vllm default (see docs/decisions/openai-backend.md).
    No real backend is constructed and no network call is made: build_backend
    is monkeypatched to capture what it was called with."""
    monkeypatch.chdir(tmp_path)
    config = _write_llm_fixture(tmp_path)

    captured = {}

    def fake_build_backend(cfg):
        captured.update(cfg)
        return _FakeBackend()

    monkeypatch.setattr("fhir_extract.eval.build_backend", fake_build_backend)

    main(config=str(config), system="llm", shots=0, constrained=False)

    assert captured["backend"] == "openai"
    assert captured["model"] == "Qwen/Qwen3-8B"

    result = json.loads(
        (tmp_path / "outputs/eval" /
         (result_tag("llm", "Qwen/Qwen3-8B", 0, False, True, "test_synthetic")
          + ".json")).read_text())
    assert result["model"] == "Qwen/Qwen3-8B"


def test_model_cli_option_overrides_inference_model(tmp_path, monkeypatch):
    """--model must win over inference.model, not be silently ignored --
    this is how the tuned-vs-base comparison gets driven."""
    monkeypatch.chdir(tmp_path)
    config = _write_llm_fixture(tmp_path)

    captured = {}

    def fake_build_backend(cfg):
        captured.update(cfg)
        return _FakeBackend()

    monkeypatch.setattr("fhir_extract.eval.build_backend", fake_build_backend)

    main(config=str(config), system="llm", model="outputs/merged/qlora-8b",
         shots=0, constrained=False)

    assert captured["model"] == "outputs/merged/qlora-8b"

    result = json.loads(
        (tmp_path / "outputs/eval" /
         (result_tag("llm", "outputs/merged/qlora-8b", 0, False, True,
                     "test_synthetic") + ".json")).read_text())
    assert result["model"] == "outputs/merged/qlora-8b"


def test_empty_split_raises_instead_of_writing_a_meaningless_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "test_synthetic.jsonl").write_text("", encoding="utf-8")
    (processed / "train.jsonl").write_text("", encoding="utf-8")

    config = tmp_path / "data.yaml"
    config.write_text(yaml.dump({"paths": {"processed": str(processed)}}),
                       encoding="utf-8")

    with pytest.raises(ValueError, match="test_synthetic.jsonl"):
        main(config=str(config), system="regex")


def test_zero_shot_eval_does_not_require_train_jsonl(tmp_path, monkeypatch):
    """With shots=0, train.jsonl should not be loaded or required to exist.
    This allows zero-shot evaluation against held-out splits without a train split."""
    monkeypatch.chdir(tmp_path)
    config = _write_llm_fixture(tmp_path)

    # Remove train.jsonl to verify it's not needed for shots=0
    train_file = tmp_path / "processed" / "train.jsonl"
    train_file.unlink()

    captured = {}

    def fake_build_backend(cfg):
        captured.update(cfg)
        return _FakeBackend()

    monkeypatch.setattr("fhir_extract.eval.build_backend", fake_build_backend)

    # Should succeed with shots=0 even without train.jsonl
    main(config=str(config), system="llm", shots=0, constrained=False)

    result = json.loads(
        (tmp_path / "outputs/eval" /
         (result_tag("llm", "Qwen/Qwen3-8B", 0, False, True, "test_synthetic")
          + ".json")).read_text())
    assert result["shots"] == 0


def test_few_shot_eval_requires_train_jsonl(tmp_path, monkeypatch):
    """With shots > 0, train.jsonl must exist; missing file should raise clear error."""
    monkeypatch.chdir(tmp_path)
    config = _write_llm_fixture(tmp_path)

    # Remove train.jsonl
    train_file = tmp_path / "processed" / "train.jsonl"
    train_file.unlink()

    def fake_build_backend(cfg):
        return _FakeBackend()

    monkeypatch.setattr("fhir_extract.eval.build_backend", fake_build_backend)

    # Should fail with shots > 0 when train.jsonl is missing
    with pytest.raises(FileNotFoundError) as exc_info:
        main(config=str(config), system="llm", shots=2, constrained=False)

    # Error message should contain the expected path and explain the requirement
    error_msg = str(exc_info.value)
    assert "train.jsonl" in error_msg
    assert "few-shot" in error_msg.lower()
    assert str(tmp_path / "processed" / "train.jsonl") in error_msg


# --- result filenames must not collide across systems --------------------

from fhir_extract.eval import result_tag


def test_tag_separates_models_evaluated_on_the_same_split():
    # The fine-tuned run overwrote the base baseline because the tag carried
    # only system/shots/split -- both were "llm_0shot_test_synthetic".
    base = result_tag("llm", "Qwen/Qwen3-8B", 0, False, True, "test_synthetic")
    ft = result_tag("llm", "fhir-lora", 0, False, False, "test_synthetic")
    assert base != ft


def test_tag_is_filesystem_safe():
    tag = result_tag("llm", "Qwen/Qwen3-8B", 0, False, True, "test_synthetic")
    assert "/" not in tag and " " not in tag


def test_tag_records_schema_hint_and_constrained():
    plain = result_tag("llm", "m", 0, False, True, "s")
    noschema = result_tag("llm", "m", 0, False, False, "s")
    constrained = result_tag("llm", "m", 0, True, True, "s")
    assert len({plain, noschema, constrained}) == 3


def test_regex_tag_needs_no_model():
    assert "regex" in result_tag("regex", None, 0, False, True, "test_synthetic")
