"""Field-level scoring for extracted clinical records.

Matching rule (documented because a fuzzy metric is only credible if its rule
is stated): text fields match if normalised strings are equal OR rapidfuzz
token_sort_ratio >= 88; numeric fields must match exactly to 1 decimal place.
"""
from collections import defaultdict
from rapidfuzz import fuzz

from .faithfulness import normalise, unanchored_facts
from .profile import ClinicalRecord, validate_as_fhir

MATCH_THRESHOLD = 88
RESOURCES = ("conditions", "medications", "allergies", "vitals", "procedures")


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
        hit = None
        for g in remaining:
            if p == g or (fuzzy and fuzz.token_sort_ratio(p, g) >= MATCH_THRESHOLD):
                hit = g
                break
        if hit is not None:
            remaining.remove(hit)
            tp += 1
    return tp, len(pred) - tp, len(remaining)


def score(pred: ClinicalRecord, gold: ClinicalRecord, note: str) -> dict:
    per_resource: dict[str, dict] = {}
    tp = fp = fn = 0
    for resource in RESOURCES:
        fuzzy = resource != "vitals"
        r_tp, r_fp, r_fn = _match(_keys(pred, resource), _keys(gold, resource), fuzzy)
        per_resource[resource] = {"tp": r_tp, "fp": r_fp, "fn": r_fn}
        tp, fp, fn = tp + r_tp, fp + r_fp, fn + r_fn

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "schema_valid": validate_as_fhir(pred) == [],
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
    n = len(rows) or 1
    return {
        "n": len(rows),
        "micro_f1": round(_f1(tp, fp, fn), 4),
        "macro_f1": round(sum(resource_f1.values()) / len(resource_f1), 4)
                    if resource_f1 else 0.0,
        "schema_validity": round(sum(bool(r["schema_valid"]) for r in rows) / n, 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in rows) / n, 4),
        "omission_rate": round(sum(r["omitted"] for r in rows) / n, 4),
        "per_resource": {k: round(v, 4) for k, v in resource_f1.items()},
    }
