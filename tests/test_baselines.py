from fhir_extract.baselines import regex_extract


def test_regex_extracts_blood_pressure():
    rec = regex_extract("Vitals: BP 142/88 mmHg, HR 78.")
    codes = {v.loinc_code: v.value for v in rec.vitals}
    assert codes["8480-6"] == 142 and codes["8462-4"] == 88

def test_regex_extracts_heart_rate():
    rec = regex_extract("HR 78 bpm")
    assert any(v.loinc_code == "8867-4" and v.value == 78 for v in rec.vitals)

def test_regex_extracts_temperature():
    rec = regex_extract("Temp 98.6 F")
    assert any(v.loinc_code == "8310-5" for v in rec.vitals)

def test_regex_finds_nothing_in_prose():
    rec = regex_extract("The patient feels unwell today.")
    assert rec.vitals == []
