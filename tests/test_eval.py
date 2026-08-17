"""Runnable proof that Ruling J's `constrained` plumbing survives end to end,
without importing vllm (system="regex" never touches LLMBaseline).
"""
import json
from pathlib import Path

import yaml

from fhir_extract.eval import main


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
