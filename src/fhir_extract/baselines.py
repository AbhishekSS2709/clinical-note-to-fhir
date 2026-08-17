"""Baselines. The regex baseline exists to be honest: it will do well on vitals
and badly elsewhere, and reporting that is the point.

Note the `constrained` flag on LLMBaseline. A 'few-shot + constrained decoding'
baseline is mandatory: comparing tuned+constrained against untuned+unconstrained
changes two variables at once and inflates the result. See spec section 8.
"""
import json
import re
from .profile import ClinicalRecord, VitalObservation, VITAL_LOINC

_PATTERNS = [
    (r"\bBP[:\s]+(\d{2,3})\s*/\s*(\d{2,3})", ("systolic_bp", "diastolic_bp"), "mmHg"),
    (r"\b(?:HR|heart rate|pulse)[:\s]+(\d{2,3})", ("heart_rate",), "/min"),
    (r"\b(?:RR|resp(?:iratory)? rate)[:\s]+(\d{1,2})", ("resp_rate",), "/min"),
    (r"\b(?:T|temp(?:erature)?)[:\s]+(\d{2,3}(?:\.\d)?)", ("temperature",), "F"),
    (r"\b(?:SpO2|O2 sat(?:uration)?)[:\s]+(\d{2,3})", ("spo2",), "%"),
    (r"\b(?:wt|weight)[:\s]+(\d{2,3}(?:\.\d)?)", ("weight",), "kg"),
    (r"\b(?:ht|height)[:\s]+(\d{2,3}(?:\.\d)?)", ("height",), "cm"),
]


def regex_extract(note: str) -> ClinicalRecord:
    vitals: list[VitalObservation] = []
    for pattern, keys, unit in _PATTERNS:
        for match in re.finditer(pattern, note, flags=re.IGNORECASE):
            for i, key in enumerate(keys):
                code, display = VITAL_LOINC[key]
                vitals.append(VitalObservation(
                    loinc_code=code, display=display,
                    value=float(match.group(i + 1)), unit=unit))
    return ClinicalRecord(vitals=vitals)


EXTRACT_INSTRUCTION = """Extract structured clinical data from the note below.

Return ONLY a JSON object with these five keys: conditions, medications,
allergies, vitals, procedures. Extract only facts explicitly stated in the note.

Note:
{note}

JSON:"""


class LLMBaseline:
    def __init__(self, model: str, shots: int = 0, constrained: bool = False,
                 examples: list[dict] | None = None):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.90)
        self.shots = shots
        self.constrained = constrained
        self.examples = examples or []
        guided = None
        if constrained:
            from vllm.sampling_params import GuidedDecodingParams
            guided = GuidedDecodingParams(json=ClinicalRecord.model_json_schema())
        self.params = SamplingParams(temperature=0.0, max_tokens=1024,
                                     guided_decoding=guided)

    def _prompt(self, note: str) -> str:
        prefix = ""
        for ex in self.examples[: self.shots]:
            prefix += EXTRACT_INSTRUCTION.format(note=ex["note"])
            prefix += json.dumps(ex["label"]) + "\n\n"
        return prefix + EXTRACT_INSTRUCTION.format(note=note)

    def extract_batch(self, notes: list[str]) -> list[tuple[ClinicalRecord, bool]]:
        outputs = self.llm.generate([self._prompt(n) for n in notes], self.params)
        records = []
        for out in outputs:
            try:
                records.append((ClinicalRecord.model_validate_json(
                    out.outputs[0].text.strip()), True))
            except Exception:
                records.append((ClinicalRecord(), False))  # unparseable == empty prediction
        return records
