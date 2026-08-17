"""Deterministic check that every labelled fact has textual support.

A generated note that dropped a fact makes its label wrong, which poisons
training. Model-free and cheap by design. See spec section 4.5.
"""
import re
from rapidfuzz import fuzz
from .profile import ClinicalRecord

FUZZ_THRESHOLD = 82

# Clinical abbreviations that count as anchors for their expansion.
ABBREVIATIONS: dict[str, list[str]] = {
    "hypertension": ["htn"],
    "diabetes mellitus": ["dm", "t2dm", "t1dm"],
    "shortness of breath": ["sob", "dyspnea"],
    "coronary artery disease": ["cad"],
    "chronic obstructive pulmonary disease": ["copd"],
    "congestive heart failure": ["chf"],
    "myocardial infarction": ["mi"],
    "urinary tract infection": ["uti"],
    "chronic kidney disease": ["ckd"],
    "atrial fibrillation": ["afib", "a-fib"],
    "gastroesophageal reflux disease": ["gerd"],
    "hydrochloride": ["hcl"],
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _text_anchored(term: str, note_norm: str) -> bool:
    term_norm = normalise(term).strip()
    if not term_norm:
        return True
    # Head word of a multi-word clinical term carries most of the signal.
    if term_norm in note_norm:
        return True
    for expansion, abbrevs in ABBREVIATIONS.items():
        if expansion in term_norm and any(
            re.search(rf"\b{re.escape(a)}\b", note_norm) for a in abbrevs
        ):
            return True
    head = term_norm.split()[0]
    if len(head) >= 5 and head in note_norm:
        return True
    return fuzz.partial_ratio(term_norm, note_norm) >= FUZZ_THRESHOLD


def _number_anchored(value: float, note: str) -> bool:
    as_int = f"{value:.0f}"
    as_one_dp = f"{value:.1f}"
    return bool(re.search(rf"\b{re.escape(as_int)}\b", note)
                or re.search(rf"\b{re.escape(as_one_dp)}\b", note))


def unanchored_facts(note: str, record: ClinicalRecord) -> list[str]:
    """Return human-readable descriptions of labelled facts absent from the note."""
    note_norm = normalise(note)
    missing: list[str] = []

    for c in record.conditions:
        if not _text_anchored(c.code_text, note_norm):
            missing.append(f"Condition: {c.code_text}")
    for m in record.medications:
        if not _text_anchored(m.medication_text, note_norm):
            missing.append(f"Medication: {m.medication_text}")
    for a in record.allergies:
        if not _text_anchored(a.substance_text, note_norm):
            missing.append(f"Allergy: {a.substance_text}")
    for v in record.vitals:
        if not _number_anchored(v.value, note):
            missing.append(f"Vital: {v.display}={v.value}")
    for p in record.procedures:
        if not _text_anchored(p.code_text, note_norm):
            missing.append(f"Procedure: {p.code_text}")
    return missing


def is_faithful(note: str, record: ClinicalRecord) -> bool:
    return not unanchored_facts(note, record)
