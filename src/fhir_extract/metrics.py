"""Field-level scoring for extracted clinical records.

Matching rule (documented because a fuzzy metric is only credible if its rule
is stated): text fields match if normalised strings are equal OR rapidfuzz
token_sort_ratio >= 88; numeric fields must match exactly to 1 decimal place.
"""
import re
from collections import defaultdict
from rapidfuzz import fuzz

from .faithfulness import normalise, unanchored_facts
from .profile import ClinicalRecord, validate_as_fhir

MATCH_THRESHOLD = 88
RESOURCES = ("conditions", "medications", "allergies", "vitals", "procedures")

_DIGIT_RE = re.compile(r"\d+")

# Opposed clinical prefixes: a word starting with one and a word starting
# with the other make the two terms clinically opposite, not variants.
_OPPOSED_PREFIXES = (
    ("hyper", "hypo"),
    ("acute", "chronic"),
    ("primary", "secondary"),
)


def _has_word_prefix(text: str, prefix: str) -> bool:
    return any(w.startswith(prefix) for w in text.split())


def _discriminators_conflict(a: str, b: str) -> bool:
    """Block fuzzy matches between clinically distinct terms.

    token_sort_ratio scores strings that differ only in a single discriminating
    token (a digit like "type 1"/"type 2" or "stage 3"/"stage 4", laterality
    like "left"/"right", or an opposed clinical prefix like "hyper"/"hypo")
    as near-identical, because that token is a small share of the string. But
    that token is exactly what makes the clinical facts different, and
    sometimes dangerously so (e.g. type 1 vs type 2 diabetes, hyperglycemia
    vs hypoglycemia). A fuzzy match must never paper over a difference in
    these tokens.
    """
    if set(_DIGIT_RE.findall(a)) != set(_DIGIT_RE.findall(b)):
        return True
    a_words, b_words = set(a.split()), set(b.split())
    if ("left" in a_words and "right" in b_words) or ("right" in a_words and "left" in b_words):
        return True
    for p1, p2 in _OPPOSED_PREFIXES:
        if _has_word_prefix(a, p1) and _has_word_prefix(b, p2):
            return True
        if _has_word_prefix(a, p2) and _has_word_prefix(b, p1):
            return True
    return False


def _keys(record: ClinicalRecord, resource: str) -> list[str]:
    if resource == "conditions":
        return [normalise(c.code_text).strip() for c in record.conditions]
    if resource == "medications":
        return [normalise(m.medication_text).strip() for m in record.medications]
    if resource == "allergies":
        return [normalise(a.substance_text).strip() for a in record.allergies]
    if resource == "vitals":
        return [f"{v.loinc_code}={v.value:.1f}" for v in record.vitals]
    if resource == "procedures":
        return [normalise(p.code_text).strip() for p in record.procedures]
    raise ValueError(resource)


def _match(pred: list[str], gold: list[str], fuzzy: bool) -> tuple[int, int, int]:
    remaining = list(gold)
    tp = 0
    for p in pred:
        best_idx = None
        best_score = -1.0
        for i, g in enumerate(remaining):
            if p == g:
                candidate_score = 100.0
            elif fuzzy and not _discriminators_conflict(p, g):
                candidate_score = fuzz.token_sort_ratio(p, g)
                if candidate_score < MATCH_THRESHOLD:
                    continue
            else:
                continue
            if candidate_score > best_score:
                best_score = candidate_score
                best_idx = i
        if best_idx is not None:
            remaining.pop(best_idx)
            tp += 1
    return tp, len(pred) - tp, len(remaining)


def score(pred: ClinicalRecord, gold: ClinicalRecord, note: str, parsed: bool = True) -> dict:
    per_resource: dict[str, dict] = {}
    tp = fp = fn = 0
    for resource in RESOURCES:
        fuzzy = resource != "vitals"
        r_tp, r_fp, r_fn = _match(_keys(pred, resource), _keys(gold, resource), fuzzy)
        per_resource[resource] = {"tp": r_tp, "fp": r_fp, "fn": r_fn}
        tp, fp, fn = tp + r_tp, fp + r_fp, fn + r_fn

    return {
        "tp": tp, "fp": fp, "fn": fn,
        # A record that never parsed as JSON is not schema-valid regardless of
        # what the fallback-empty ClinicalRecord() would otherwise validate as.
        "schema_valid": parsed and validate_as_fhir(pred) == [],
        # A predicted fact with no anchor in the note is a hallucination.
        "hallucinated": len(unanchored_facts(note, pred)),
        "omitted": fn,
        "per_resource": per_resource,
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def aggregate(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)

    per_resource: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        for name, counts in r.get("per_resource", {}).items():
            for k in ("tp", "fp", "fn"):
                per_resource[name][k] += counts[k]

    resource_f1 = {n: _f1(c["tp"], c["fp"], c["fn"]) for n, c in per_resource.items()}
    # Resource types with no gold and no predictions are absent from this split,
    # not failures; including them would drag a perfect model's macro-F1 down.
    nonempty_f1 = [resource_f1[n] for n, c in per_resource.items()
                   if c["tp"] + c["fp"] + c["fn"] > 0]
    n = len(rows) or 1
    return {
        "n": len(rows),
        "micro_f1": round(_f1(tp, fp, fn), 4),
        "macro_f1": round(sum(nonempty_f1) / len(nonempty_f1), 4)
                    if nonempty_f1 else 0.0,
        "schema_validity": round(sum(bool(r["schema_valid"]) for r in rows) / n, 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in rows) / n, 4),
        "omission_rate": round(sum(r["omitted"] for r in rows) / n, 4),
        "per_resource": {k: round(v, 4) for k, v in resource_f1.items()},
    }
