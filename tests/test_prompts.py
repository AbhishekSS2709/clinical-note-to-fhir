import random
from fhir_extract.profile import ClinicalRecord, Condition
from fhir_extract.synthea import EncounterRecord
from fhir_extract.prompts import build_prompt, DOC_TYPES, STYLES, NOISE


def _enc():
    return EncounterRecord(patient_id="p", encounter_id="e",
                           encounter_date="2020-01-01", age=61, sex="female",
                           record=ClinicalRecord())


def test_matrix_has_expected_dimensions():
    assert len(DOC_TYPES) == 6 and len(STYLES) == 4 and len(NOISE) == 5


def test_prompt_contains_every_label_fact():
    sub = ClinicalRecord(conditions=[Condition(code_text="Acute bronchitis",
                                               clinical_status="active")])
    prompt, meta = build_prompt(_enc(), sub, random.Random(0))
    assert "Acute bronchitis" in prompt


def test_prompt_metadata_records_the_variant():
    _, meta = build_prompt(_enc(), ClinicalRecord(), random.Random(0))
    assert meta["doc_type"] in DOC_TYPES
    assert meta["style"] in STYLES
    assert meta["noise"] in NOISE


def test_variants_differ_across_seeds():
    metas = {tuple(build_prompt(_enc(), ClinicalRecord(), random.Random(s))[1].values())
             for s in range(40)}
    assert len(metas) > 5, "prompt matrix must actually vary"


def test_prompt_forbids_restating_field_names():
    prompt, _ = build_prompt(_enc(), ClinicalRecord(), random.Random(0))
    assert "condition is active" in prompt.lower()  # the forbidden example itself
    assert "never state field names or metadata verbatim" in prompt.lower()

def test_prompt_requires_stripping_semantic_tag_when_mentioning_terms():
    prompt, _ = build_prompt(_enc(), ClinicalRecord(), random.Random(0))
    assert "(disorder)" in prompt.lower()  # cited as the example to strip


def test_encounter_rng_is_order_independent():
    """A resumed run must reproduce a fresh run: same seed + id -> same stream."""
    from fhir_extract.generate import _encounter_rng
    a = _encounter_rng(42, "enc-abc")
    b = _encounter_rng(42, "enc-abc")
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    c = _encounter_rng(42, "enc-xyz")
    assert _encounter_rng(42, "enc-abc").random() != c.random()
