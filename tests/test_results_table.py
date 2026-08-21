import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from results_table import classify


def test_classify_separates_the_two_adapters():
    # Both are 0-shot LLM runs on the same split; only the model tells them
    # apart, which is exactly the collision that overwrote a result file.
    bf16 = {"system": "llm", "model": "fhir-lora", "shots": 0}
    qlora = {"system": "llm", "model": "fhir-qlora", "shots": 0}
    assert classify(bf16) != classify(qlora)


def test_classify_distinguishes_constrained_from_plain():
    plain = {"system": "llm", "model": "qwen3-8b-base", "shots": 0}
    constrained = {"system": "llm", "model": "qwen3-8b-base", "shots": 0,
                   "constrained": True}
    assert classify(plain) != classify(constrained)


def test_classify_distinguishes_shot_counts():
    zero = {"system": "llm", "model": "qwen3-8b-base", "shots": 0}
    five = {"system": "llm", "model": "qwen3-8b-base", "shots": 5}
    assert classify(zero) != classify(five)


def test_classify_regex_needs_no_model():
    assert classify({"system": "regex", "model": None, "shots": 0}) == "regex"


def test_classify_recognises_the_v2_adapter():
    # "fhir-v2" matched none of the adapter patterns and fell through to the
    # base-model row, silently overwriting the base result in the table.
    v2 = {"system": "llm", "model": "fhir-v2", "shots": 0}
    base = {"system": "llm", "model": "qwen3-8b-base", "shots": 0}
    assert classify(v2) != classify(base)
    assert classify(v2) == "ft-elmtex"


def test_classify_separates_v2_checkpoints():
    # All four v2 checkpoints mapped to one key, so a table containing the
    # data-efficiency curve kept only the last row -- the same collision as
    # the result filenames, third occurrence.
    keys = {classify({"system": "llm", "model": m, "shots": 0})
            for m in ("fhir-v2-100", "fhir-v2-200", "fhir-v2-300", "fhir-v2-425")}
    assert len(keys) == 4
