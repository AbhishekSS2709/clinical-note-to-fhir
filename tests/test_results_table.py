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
