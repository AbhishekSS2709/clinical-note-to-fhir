"""Prompt rotation matrix. Diversity is the risk here, not quality -- a single
prompt at default temperature yields thousands of notes in one voice and the
student overfits to it. See spec section 4.4.
"""
import random
from .profile import ClinicalRecord
from .synthea import EncounterRecord

DOC_TYPES = [
    "SOAP progress note", "discharge summary", "emergency department note",
    "referral letter", "telephone encounter note", "history and physical",
]
STYLES = [
    "terse and abbreviation-heavy, as a busy clinician types",
    "verbose narrative prose",
    "bulleted and heavily structured",
    "a dictated transcript with run-on sentences",
]
NOISE = [
    "clean and well-formatted",
    "containing a few realistic typos",
    "using inconsistent units (pounds and kilograms, Fahrenheit and Celsius)",
    "with some copy-forward duplication from a previous note",
    "with heavy use of negation for pertinent negatives",
]

_TEMPLATE = """You are writing a realistic clinical note for a training corpus.

Patient: {age}-year-old {sex}. Encounter date: {date}.

You MUST mention every one of these clinical facts somewhere in the note:
{facts}

Rules:
- Mention EVERY fact listed above. Omitting one makes the note unusable.
- Do NOT invent any additional diagnoses, medications, allergies, vital
  measurements, or procedures beyond those listed. Filler such as chief
  complaint narrative, exam prose, and disposition is encouraged.
- Write like a clinician documenting a patient, not like a system summarising
  a record. Never state field names or metadata verbatim -- no "condition is active",
  no "onset <date>". Work dates into the prose naturally ("since 2019",
  "diagnosed three years ago") or leave them out.
- When mentioning a condition or procedure, drop any parenthetical SNOMED
  tag -- write "chronic sinusitis", never "chronic sinusitis (disorder)".
- Write it as a {doc_type}, {style}, {noise}.
- Output ONLY the note text. No preamble, no JSON, no commentary.

Note:"""


def _facts(subset: ClinicalRecord) -> str:
    lines: list[str] = []
    for c in subset.conditions:
        onset = f" (onset {c.onset_date})" if c.onset_date else ""
        lines.append(f"- Condition: {c.code_text}, status {c.clinical_status}{onset}")
    for m in subset.medications:
        d = m.dosage
        parts = [p for p in [
            f"{d.dose}{d.unit}" if d.dose and d.unit else None,
            d.route, d.frequency] if p]
        detail = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- Medication: {m.medication_text}{detail}")
    for a in subset.allergies:
        rx = f", reaction: {', '.join(a.manifestation)}" if a.manifestation else ""
        lines.append(f"- Allergy: {a.substance_text}{rx}")
    for v in subset.vitals:
        lines.append(f"- Vital sign: {v.display} = {v.value} {v.unit}")
    for p in subset.procedures:
        when = f" on {p.performed_date}" if p.performed_date else ""
        lines.append(f"- Procedure: {p.code_text}{when}")
    return "\n".join(lines) if lines else "- (no specific findings; routine visit)"


def build_prompt(enc: EncounterRecord, subset: ClinicalRecord,
                 rng: random.Random) -> tuple[str, dict]:
    meta = {
        "doc_type": rng.choice(DOC_TYPES),
        "style": rng.choice(STYLES),
        "noise": rng.choice(NOISE),
    }
    prompt = _TEMPLATE.format(
        age=enc.age, sex=enc.sex, date=enc.encounter_date,
        facts=_facts(subset), **meta,
    )
    return prompt, meta
