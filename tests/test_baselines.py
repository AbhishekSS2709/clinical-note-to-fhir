import pytest

from fhir_extract.baselines import LLMBaseline, regex_extract, parse_record


def test_regex_extracts_blood_pressure():
    rec = regex_extract("Vitals: BP 142/88 mmHg, HR 78.")
    codes = {v.loinc_code: v.value for v in rec.vitals}
    assert codes["8480-6"] == 142 and codes["8462-4"] == 88

def test_regex_extracts_heart_rate():
    rec = regex_extract("HR 78 bpm")
    assert any(v.loinc_code == "8867-4" and v.value == 78 for v in rec.vitals)

def test_regex_extracts_temperature():
    rec = regex_extract("Temp 98.6 F")
    assert any(v.loinc_code == "8310-5" for v in rec.vitals)

def test_regex_finds_nothing_in_prose():
    rec = regex_extract("The patient feels unwell today.")
    assert rec.vitals == []


def test_parse_record_accepts_plain_json():
    record, parsed = parse_record('{"conditions": [], "medications": [], '
                                   '"allergies": [], "vitals": [], "procedures": []}')
    assert parsed is True
    assert record.conditions == []


def test_parse_record_strips_markdown_json_fence():
    text = ('```json\n{"conditions": [], "medications": [], "allergies": [], '
            '"vitals": [], "procedures": []}\n```')
    record, parsed = parse_record(text)
    assert parsed is True
    assert record.medications == []


def test_parse_record_strips_bare_fence_without_json_tag():
    text = ('```\n{"conditions": [], "medications": [], "allergies": [], '
            '"vitals": [], "procedures": []}\n```')
    record, parsed = parse_record(text)
    assert parsed is True


def test_parse_record_unparseable_text_returns_empty_record_and_false():
    record, parsed = parse_record("not json at all")
    assert parsed is False
    assert record.conditions == [] and record.medications == []


def test_llm_baseline_raises_clear_error_with_neither_backend_nor_config():
    """No hardcoded vllm fallback: LLMBaseline must not silently default to
    a backend that isn't installed. See docs/decisions/openai-backend.md."""
    with pytest.raises(ValueError, match="backend"):
        LLMBaseline(model="Qwen/Qwen3-8B")
