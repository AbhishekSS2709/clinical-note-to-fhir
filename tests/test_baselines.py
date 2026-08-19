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


# --- reasoning-block tolerance -------------------------------------------

def test_parse_record_accepts_a_leading_empty_think_block():
    # Qwen3's chat template injects <think></think> INSIDE the assistant turn,
    # so a model fine-tuned through that template emits it before the JSON.
    # Without this, every fine-tuned prediction fails to parse and the model
    # scores 0.0 while actually working.
    rec, ok = parse_record('<think>\n\n</think>\n\n{"conditions": [], '
                           '"medications": [], "allergies": [], "vitals": [], '
                           '"procedures": []}')
    assert ok and rec.conditions == []


def test_parse_record_accepts_a_think_block_with_content():
    rec, ok = parse_record('<think>The note mentions asthma.</think>\n'
                           '{"conditions": [{"code_text": "asthma", '
                           '"clinical_status": "active"}]}')
    assert ok and rec.conditions[0].code_text == "asthma"


def test_parse_record_accepts_a_think_block_wrapped_in_a_code_fence():
    rec, ok = parse_record('<think></think>\n```json\n{"conditions": []}\n```')
    assert ok


def test_parse_record_still_reports_failure_on_genuinely_broken_output():
    # Leniency must not turn unparseable output into a silent empty record.
    rec, ok = parse_record("<think>I cannot answer</think> sorry, no JSON here")
    assert not ok and rec.conditions == []


# --- baseline fairness: the schema must be in the prompt ------------------

class _StubBackend:
    """Minimal LLMBackend: records prompts, returns canned replies."""

    def __init__(self, replies):
        self.replies = replies
        self.prompts = []

    def complete(self, prompts, **kw):
        self.prompts.extend(prompts)
        return list(self.replies) + [""] * (len(prompts) - len(self.replies))


def test_baseline_prompt_states_the_schema_by_default():
    # The fine-tuned model learns the field structure from 17k examples. A
    # baseline that is only told the five key NAMES invents its own shape
    # (vitals as a dict, conditions as strings) and scores 0.0 -- a strawman.
    b = LLMBaseline("m", backend=_StubBackend([]))
    prompt = b._prompt("Patient is well.")
    assert "loinc_code" in prompt and "clinical_status" in prompt


def test_schema_hint_can_be_disabled_to_match_the_training_prompt():
    # The fine-tuned model must be evaluated on the prompt it was trained on,
    # which carries no schema block.
    from fhir_extract.baselines import EXTRACT_INSTRUCTION
    b = LLMBaseline("m", backend=_StubBackend([]), schema_hint=False)
    assert b._prompt("Patient is well.") == \
        EXTRACT_INSTRUCTION.format(note="Patient is well.")


def test_schema_hint_describes_every_profiled_resource():
    b = LLMBaseline("m", backend=_StubBackend([]))
    prompt = b._prompt("n")
    for key in ("conditions", "medications", "allergies", "vitals", "procedures"):
        assert key in prompt
