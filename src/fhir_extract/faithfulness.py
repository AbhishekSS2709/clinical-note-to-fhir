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


# Trailing SNOMED semantic tags. Labels read "Chronic sinusitis (disorder)"
# while notes say "chronic sinusitis" -- the tag is not part of what a
# clinician would ever write, and dragging it into the fuzzy comparison
# depresses the match score. Only a *trailing* parenthetical exactly matching
# one of these is stripped; mid-string parenthetical content is untouched.
_SEMANTIC_TAGS = frozenset({
    "disorder", "finding", "procedure", "situation",
    "regime/therapy", "substance", "morphologic abnormality",
})
_TRAILING_TAG_RE = re.compile(r"^(.*)\s*\(([a-z/ ]+)\)\s*$", re.IGNORECASE)


def _strip_semantic_tag(text: str) -> str:
    match = _TRAILING_TAG_RE.match(text.strip())
    if match and match.group(2).strip().lower() in _SEMANTIC_TAGS:
        return match.group(1).strip()
    return text


def normalise(text: str) -> str:
    text = _strip_semantic_tag(text)
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _text_anchored(term: str, note_norm: str) -> bool:
    term_norm = normalise(term).strip()
    if not term_norm:
        # An empty or whitespace-only term is unverifiable, not verified.
        # Fail closed: an unanchorable fact must be reported as unanchored,
        # not silently counted as anchored.
        return False
    # Head word of a multi-word clinical term carries most of the signal.
    if term_norm in note_norm:
        return True
    for expansion, abbrevs in ABBREVIATIONS.items():
        if expansion in term_norm and any(
            re.search(rf"\b{re.escape(a)}\b", note_norm) for a in abbrevs
        ):
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


# Common drug names/stems. Scope: this list is deliberately small and generic
# (widely prescribed drugs across the major classes a Synthea-derived note
# would plausibly mention) rather than an exhaustive formulary, because
# invented_facts must be high-precision/low-recall -- a false positive here
# only costs one training pair, but a miss lets an invented drug silently
# poison a label. Deterministic substring matching, no external drug DB.
_COMMON_DRUGS = (
    "metformin", "lisinopril", "amoxicillin", "clavulanic acid", "atorvastatin",
    "albuterol", "insulin glargine", "insulin lispro", "insulin aspart",
    "insulin detemir", "insulin", "warfarin", "levothyroxine", "amlodipine",
    "metoprolol", "omeprazole", "losartan", "gabapentin", "hydrochlorothiazide",
    "simvastatin", "sertraline", "ibuprofen", "acetaminophen", "aspirin",
    "clopidogrel", "prednisone", "prednisolone", "furosemide", "citalopram",
    "escitalopram", "trazodone", "tramadol", "montelukast", "pantoprazole",
    "rosuvastatin", "duloxetine", "venlafaxine", "bupropion", "fluoxetine",
    "paroxetine", "alprazolam", "lorazepam", "clonazepam", "diazepam",
    "zolpidem", "cyclobenzaprine", "azithromycin", "ciprofloxacin",
    "doxycycline", "cephalexin", "clindamycin", "levofloxacin", "metronidazole",
    "nitrofurantoin", "sulfamethoxazole", "trimethoprim", "penicillin",
    "vancomycin", "meropenem", "ceftriaxone", "fluconazole", "acyclovir",
    "valacyclovir", "oseltamivir", "hydrocodone", "oxycodone", "morphine",
    "fentanyl", "codeine", "naproxen", "celecoxib", "meloxicam", "diclofenac",
    "indomethacin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin",
    "heparin", "digoxin", "amiodarone", "diltiazem", "verapamil", "carvedilol",
    "atenolol", "propranolol", "spironolactone", "hydralazine", "isosorbide",
    "nitroglycerin", "clonidine", "doxazosin", "tamsulosin", "finasteride",
    "sildenafil", "tadalafil", "methotrexate", "hydroxychloroquine",
    "sulfasalazine", "azathioprine", "adalimumab", "etanercept", "infliximab",
    "dexamethasone", "hydrocortisone", "fluticasone", "budesonide",
    "mometasone", "tiotropium", "ipratropium", "salmeterol", "formoterol",
    "cetirizine", "loratadine", "diphenhydramine", "fexofenadine", "ranitidine",
    "famotidine", "sucralfate", "misoprostol", "loperamide", "ondansetron",
    "promethazine", "metoclopramide", "docusate", "senna", "bisacodyl",
    "lactulose", "glipizide", "glyburide", "glimepiride", "sitagliptin",
    "empagliflozin", "canagliflozin", "dapagliflozin", "liraglutide",
    "semaglutide", "pioglitazone", "levetiracetam", "phenytoin",
    "carbamazepine", "valproate", "lamotrigine", "topiramate", "pregabalin",
    "baclofen", "tizanidine", "methocarbamol", "quetiapine", "risperidone",
    "olanzapine", "aripiprazole", "haloperidol", "lithium", "buspirone",
    "hydroxyzine", "erythromycin", "gentamicin", "tobramycin", "linezolid",
    "rifampin", "isoniazid", "ethambutol", "pyrazinamide",
)

_BP_RE = re.compile(r"\bBP\s*:?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", re.IGNORECASE)


def invented_facts(note: str, record: ClinicalRecord) -> list[str]:
    """Return human-readable descriptions of facts present in the note but
    absent from the label -- i.e. facts the generator invented.

    Deterministic and model-free, like unanchored_facts. Deliberately
    high-precision/low-recall: a false positive merely drops one usable
    training pair, but a false negative trains the model to ignore a fact
    plainly stated in the text, which silently corrupts the dataset.
    Scope is therefore narrow -- a curated common-drug list plus a `BP
    nnn/nn` numeric pattern -- rather than a general entity extractor.
    """
    note_norm = normalise(note)
    label_meds_norm = [normalise(m.medication_text) for m in record.medications]
    invented: list[str] = []

    for drug in _COMMON_DRUGS:
        drug_norm = normalise(drug)
        if not re.search(rf"\b{re.escape(drug_norm)}\b", note_norm):
            continue
        if any(drug_norm in lm or fuzz.partial_ratio(drug_norm, lm) >= FUZZ_THRESHOLD
               for lm in label_meds_norm):
            continue
        invented.append(f"Medication (invented): {drug}")

    label_values = {round(v.value, 1) for v in record.vitals}
    for m in _BP_RE.finditer(note):
        sys_val, dia_val = float(m.group(1)), float(m.group(2))
        if round(sys_val, 1) in label_values or round(dia_val, 1) in label_values:
            continue
        invented.append(f"Vital (invented): BP {m.group(1)}/{m.group(2)}")

    return invented


def is_clean(note: str, record: ClinicalRecord) -> bool:
    return not unanchored_facts(note, record) and not invented_facts(note, record)
