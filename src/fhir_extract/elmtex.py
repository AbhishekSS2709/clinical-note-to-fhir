"""Convert the ELMTEX clinical-report corpus into our label format.

ELMTEX (arXiv:2502.05638, CC-BY-4.0, doi:10.5281/zenodo.14793810) annotates
real clinical case reports from PubMed Central with 15 free-text categories.
It is the external-validity check for a model trained entirely on Synthea-
derived synthetic notes: the reports are human-written, so performance here
measures the distribution shift the synthetic corpus cannot.

Only 3 of our 5 resource types are covered. ELMTEX annotates no vital signs
and no allergies, so those must be EXCLUDED from scoring rather than scored
against empty gold -- the reports themselves do mention vitals, and a model
that correctly extracts them would otherwise be charged a false positive for
every one.

Labels are free-text names, not coded structures: there are no clinical
statuses, dosages or dates to match, so an ELMTEX score is name-level and is
not comparable to the synthetic-split numbers.
"""
import json
from pathlib import Path

import typer

from .profile import ClinicalRecord, Condition, MedicationStatement, Procedure

app = typer.Typer()

# Which ELMTEX categories feed which resource type.
#
# `medical_surgical_history` is deliberately absent: it mixes past diagnoses
# with past operations, so assigning it to conditions or to procedures invents
# gold facts, and assigning it to both double-counts them.
ELMTEX_FIELDS = {
    "conditions": ["diagnosis", "comorbidities"],
    "medications": ["pharmacological_therapy"],
    "procedures": ["interventional_therapy", "diagnostic_techniques_procedures"],
}

# Resource types an ELMTEX comparison may score. Pass to metrics.score().
ELMTEX_RESOURCES = ("conditions", "medications", "procedures")

_NULL = {"", "n/a", "na", "none", "not applicable"}


def split_entries(value: str) -> list[str]:
    """Split one ELMTEX field into individual facts.

    Entries are semicolon-delimited. "N/A" is ELMTEX's null marker and must
    become an empty list -- carried through as a string it becomes a phantom
    gold fact that no model can ever match, depressing recall everywhere.
    """
    if not isinstance(value, str) or value.strip().lower() in _NULL:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def to_record(summary: dict) -> ClinicalRecord:
    """Build a ClinicalRecord from one ELMTEX `summary` object."""
    conditions, medications, procedures = [], [], []
    for field in ELMTEX_FIELDS["conditions"]:
        conditions += [Condition(code_text=t, clinical_status="active")
                       for t in split_entries(summary.get(field, ""))]
    for field in ELMTEX_FIELDS["medications"]:
        medications += [MedicationStatement(medication_text=t)
                        for t in split_entries(summary.get(field, ""))]
    for field in ELMTEX_FIELDS["procedures"]:
        procedures += [Procedure(code_text=t)
                       for t in split_entries(summary.get(field, ""))]
    # vitals and allergies stay empty: ELMTEX does not annotate them.
    return ClinicalRecord(conditions=conditions, medications=medications,
                          procedures=procedures)


def split_rows(rows: list[dict], val_rows: int,
               seed: int) -> tuple[list[dict], list[dict]]:
    """Patient-level train/val split.

    ELMTEX groups several reports under one `patient_uid`, so splitting by row
    would leak a patient's phrasing across the boundary exactly as it would on
    the synthetic corpus.
    """
    import random
    from collections import defaultdict

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    val: list[dict] = []
    held: set[str] = set()
    for patient in patients:
        if len(val) >= val_rows:
            break
        held.add(patient)
        val.extend(by_patient[patient])
    train = [r for r in rows if r["patient_id"] not in held]
    return train, val


@app.command()
def main(source: str, out: str = "", limit: int = 0,
         train_out: str = "", val_out: str = "", val_rows: int = 400,
         seed: int = 42) -> None:
    """Write ELMTEX records as note/label JSONL matching our splits.

    With --train-out/--val-out, emits a patient-disjoint train/val pair for
    fine-tuning on real reports instead of a single evaluation file.
    """
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    if limit:
        records = records[:limit]
    rows = []
    for rec in records:
        label = to_record(rec.get("summary") or {})
        # A report with no fact in any covered type cannot discriminate
        # between systems; it only adds precision-only rows.
        if not (label.conditions or label.medications or label.procedures):
            continue
        rows.append({
            "patient_id": str(rec.get("patient_uid") or rec.get("patient_id")),
            "encounter_id": str(rec.get("PMID")),
            "note": rec["report"],
            "label": label.model_dump(),
        })

    def dump(path: str, subset: list[dict]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for r in subset:
                fh.write(json.dumps(r) + "\n")
        typer.echo(f"wrote {len(subset)} rows to {target}")

    if train_out:
        train, val = split_rows(rows, val_rows, seed)
        dump(train_out, train)
        dump(val_out or str(Path(train_out).with_name("val.jsonl")), val)
    else:
        dump(out, rows)
    typer.echo(f"kept {len(rows)}/{len(records)} records")


if __name__ == "__main__":
    app()
