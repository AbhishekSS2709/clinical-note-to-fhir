"""Count how many ELMTEX categories map onto our five FHIR resource types."""
import json, sys
from collections import Counter
from pathlib import Path

TARGETS = {
    "conditions": ["diagnosis", "medical history", "comorbid", "condition"],
    "medications": ["medication", "drug", "treatment", "therapy"],
    "allergies": ["allergy", "allergies", "adverse"],
    "vitals": ["vital", "blood pressure", "temperature", "heart rate"],
    "procedures": ["procedure", "surgery", "operation", "intervention"],
}

def main(path: str) -> None:
    keys = Counter()
    for f in Path(path).rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        records = data if isinstance(data, list) else [data]
        for rec in records:
            if isinstance(rec, dict):
                keys.update(rec.keys())

    print(f"Found {len(keys)} distinct annotation keys\n")
    for k, n in keys.most_common():
        print(f"  {n:6d}  {k}")

    print("\n--- Mapping to our profile ---")
    mapped = 0
    for target, needles in TARGETS.items():
        hits = [k for k in keys if any(n in k.lower() for n in needles)]
        status = "MAPS" if hits else "NO MATCH"
        if hits:
            mapped += 1
        print(f"{target:14s} {status:9s} {hits}")
    print(f"\nRESULT: {mapped}/5 resource types covered.")
    print("Decision rule: >=3 -> build the mapper; <3 -> fall back to MTSamples.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/elmtex")
