"""Choose which facts a realistic clinical note would actually mention.

Synthea bundles are complete patient records. Real notes are not. Labelling a
generated note with the FULL bundle would train the model to emit facts absent
from the text -- i.e. to hallucinate. So we select a plausible subset FIRST,
and that subset becomes the label. See spec section 4.3.
"""
import random
import re
from .profile import ClinicalRecord
from .synthea import EncounterRecord

# Administrative/billing/screening procedures Synthea attaches to every
# encounter regardless of clinical relevance. No clinician restates these in
# a note narrative, so they must never enter the label -- otherwise a correct
# omission gets counted as a generation failure. Matched as a substring
# anywhere in the term (case-insensitive), not a full-term match, so multi-word
# variants ("Screening for domestic abuse") are still caught.
_ADMIN_KEYWORDS = (
    "screening", "screening for", "assessment of", "assessment using",
    "education", "reconciliation", "review due", "referral for",
    "counseling", "counselling", "evaluation of", "health and social care",
    "questionnaire", "notification", "certification", "interview",
)

# Social-determinant-of-health findings (housing, employment, stress, etc.)
# that Synthea models as clinical Conditions but that are not what a
# clinician documents as a diagnosis/finding in a note the way "obesity" is.
_SOCIAL_DETERMINANT_KEYWORDS = (
    "stress", "employment", "education", "housing", "social",
    "criminal", "refugee", "transport", "income", "literacy",
)

_TRAILING_TAG_RE = re.compile(r"\(([a-z]+)\)\s*$")


def is_narratable(code_text: str, resource: str) -> bool:
    """True if a clinician would plausibly write this concept in a note.

    Filters out Synthea's billing/workflow scaffolding (administrative
    procedures, SNOMED "situation" codes, social-determinant findings, and
    our own parser's "unknown" fallback) so it never enters a label -- a
    generated note correctly omitting a code no one narrates should not be
    counted as a faithfulness failure.
    """
    if not code_text or not code_text.strip() or code_text.strip().lower() == "unknown":
        return False
    text = code_text.lower()
    tag_match = _TRAILING_TAG_RE.search(text)
    tag = tag_match.group(1) if tag_match else None
    if tag == "situation":
        return False
    if resource == "Procedure" and any(kw in text for kw in _ADMIN_KEYWORDS):
        return False
    if resource == "Condition" and tag == "finding" and any(
            kw in text for kw in _SOCIAL_DETERMINANT_KEYWORDS):
        return False
    return True

# Retention rates chosen to mimic note-writing behaviour: clinicians restate
# active problems and current meds, but rarely the full historical list.
KEEP_CONDITIONS = (0.4, 0.8)   # fraction range
KEEP_MEDICATIONS = (0.5, 1.0)
KEEP_PROCEDURES = (0.5, 1.0)
ALLERGY_MENTION_PROB = 0.7     # notes often but not always restate allergies
SMALL_RECORD_THRESHOLD = 3     # at/below this, keep everything


def _sample(items: list, rng: random.Random, rate_range: tuple[float, float]) -> list:
    if len(items) <= SMALL_RECORD_THRESHOLD:
        return list(items)
    rate = rng.uniform(*rate_range)
    k = max(1, round(len(items) * rate))
    return rng.sample(items, k)


def select_subset(enc: EncounterRecord, rng: random.Random) -> ClinicalRecord:
    src = enc.record
    # Drop non-narratable concepts (administrative procedures, "situation"
    # codes, social-determinant findings, parser fallbacks) before sampling,
    # so they can never end up in the label.
    conditions = [c for c in src.conditions if is_narratable(c.code_text, "Condition")]
    medications = [m for m in src.medications
                   if is_narratable(m.medication_text, "MedicationStatement")]
    procedures = [p for p in src.procedures if is_narratable(p.code_text, "Procedure")]
    allergies = [a for a in src.allergies
                 if is_narratable(a.substance_text, "AllergyIntolerance")]

    sub = ClinicalRecord(
        conditions=_sample(conditions, rng, KEEP_CONDITIONS),
        medications=_sample(medications, rng, KEEP_MEDICATIONS),
        procedures=_sample(procedures, rng, KEEP_PROCEDURES),
        # Vitals are measured AT this encounter; if the note reports vitals it
        # reports the set. Keep all.
        vitals=list(src.vitals),
        allergies=[a for a in allergies if rng.random() < ALLERGY_MENTION_PROB],
    )
    if not any([sub.conditions, sub.medications, sub.allergies,
                sub.vitals, sub.procedures]):
        # Never emit an empty label; fall back to one condition, else vitals,
        # else one medication, else one allergy.
        if conditions:
            sub.conditions = [conditions[0]]
        elif src.vitals:
            sub.vitals = list(src.vitals)
        elif medications:
            sub.medications = [medications[0]]
        elif allergies:
            sub.allergies = [allergies[0]]
    return sub
