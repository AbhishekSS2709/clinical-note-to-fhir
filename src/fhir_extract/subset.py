"""Choose which facts a realistic clinical note would actually mention.

Synthea bundles are complete patient records. Real notes are not. Labelling a
generated note with the FULL bundle would train the model to emit facts absent
from the text -- i.e. to hallucinate. So we select a plausible subset FIRST,
and that subset becomes the label. See spec section 4.3.
"""
import random
from .profile import ClinicalRecord
from .synthea import EncounterRecord

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
    sub = ClinicalRecord(
        conditions=_sample(src.conditions, rng, KEEP_CONDITIONS),
        medications=_sample(src.medications, rng, KEEP_MEDICATIONS),
        procedures=_sample(src.procedures, rng, KEEP_PROCEDURES),
        # Vitals are measured AT this encounter; if the note reports vitals it
        # reports the set. Keep all.
        vitals=list(src.vitals),
        allergies=[a for a in src.allergies if rng.random() < ALLERGY_MENTION_PROB],
    )
    if not any([sub.conditions, sub.medications, sub.allergies,
                sub.vitals, sub.procedures]):
        # Never emit an empty label; fall back to one condition, else vitals,
        # else one medication, else one allergy.
        if src.conditions:
            sub.conditions = [src.conditions[0]]
        elif src.vitals:
            sub.vitals = list(src.vitals)
        elif src.medications:
            sub.medications = [src.medications[0]]
        elif src.allergies:
            sub.allergies = [src.allergies[0]]
    return sub
