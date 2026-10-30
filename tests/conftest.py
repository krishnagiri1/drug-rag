import json

import pytest

DRUGS = [
    ("Warfarin Sodium", "Coumadin", {
        "boxed_warning": "WARNING: BLEEDING RISK. Warfarin can cause major or fatal bleeding. Perform regular monitoring of INR.",
        "indications_and_usage": "Warfarin is indicated for prophylaxis and treatment of venous thrombosis and pulmonary embolism.",
        "contraindications": "Warfarin is contraindicated in pregnancy except in women with mechanical heart valves, "
                             "and in patients with hemorrhagic tendencies or blood dyscrasias.",
        "drug_interactions": "Concomitant use with aspirin, NSAIDs, and certain antibiotics increases bleeding risk. "
                             "CYP2C9 inhibitors such as fluconazole increase warfarin exposure.",
        "adverse_reactions": "The most common adverse reactions are fatal and nonfatal hemorrhage, necrosis, and calciphylaxis.",
    }),
    ("Metformin Hydrochloride", "Glucophage", {
        "boxed_warning": "WARNING: LACTIC ACIDOSIS. Postmarketing cases of metformin-associated lactic acidosis have resulted in death.",
        "indications_and_usage": "Metformin is indicated as an adjunct to diet and exercise to improve glycemic control "
                                 "in adults and pediatric patients 10 years and older with type 2 diabetes mellitus.",
        "contraindications": "Metformin is contraindicated in patients with severe renal impairment (eGFR below 30 mL/min/1.73 m2) "
                             "and in acute or chronic metabolic acidosis, including diabetic ketoacidosis.",
        "dosage_and_administration": "Start at 500 mg orally twice a day or 850 mg once a day with meals. "
                                     "The maximum recommended daily dose is 2550 mg.",
        "adverse_reactions": "The most common adverse reactions are diarrhea, nausea, vomiting, flatulence, and asthenia.",
    }),
    ("Simvastatin", "Zocor", {
        "indications_and_usage": "Simvastatin is an HMG-CoA reductase inhibitor indicated to reduce the risk of total mortality "
                                 "and LDL cholesterol in adults with hyperlipidemia.",
        "contraindications": "Simvastatin is contraindicated with strong CYP3A4 inhibitors, gemfibrozil, cyclosporine, "
                             "and danazol, and in active liver disease.",
        "drug_interactions": "Strong CYP3A4 inhibitors such as itraconazole and clarithromycin increase the risk of "
                             "myopathy and rhabdomyolysis. Do not exceed 10 mg daily with verapamil or diltiazem.",
        "dosage_and_administration": "The usual dosage range is 5 mg to 40 mg once daily in the evening.",
        "overdosage": "Supportive measures should be taken. The dialyzability of simvastatin is not known.",
    }),
    ("Amoxicillin", "Amoxil", {
        "indications_and_usage": "Amoxicillin is a penicillin-class antibacterial indicated for infections of the ear, nose, "
                                 "throat, genitourinary tract, skin, and lower respiratory tract.",
        "contraindications": "Amoxicillin is contraindicated in patients with a history of serious hypersensitivity "
                             "reactions such as anaphylaxis to amoxicillin or other beta-lactams.",
        "adverse_reactions": "The most common adverse reactions are diarrhea, rash, vomiting, and nausea.",
        "dosage_and_administration": "Adults: 500 mg every 12 hours or 250 mg every 8 hours depending on the infection.",
        "mechanism_of_action": "Amoxicillin inhibits bacterial cell wall synthesis by binding penicillin-binding proteins.",
    }),
]


def make_labels(repeat_filler: int = 6):
    labels = []
    for i, (generic, brand, sections) in enumerate(DRUGS):
        secs = {}
        for k, v in sections.items():
            filler = " ".join(f"Additional {k.replace('_', ' ')} detail number {j} for {generic}." for j in range(repeat_filler))
            secs[k] = v + " " + filler
        labels.append({"set_id": f"set-{i}", "label_id": f"id-{i}", "effective_time": "20250101",
                       "generic_name": generic, "brand_name": brand, "manufacturer": "Test Pharma",
                       "route": "ORAL", "sections": secs})
    return labels


@pytest.fixture
def labels():
    return make_labels()


@pytest.fixture
def built(tmp_path, labels):
    """Labels on disk, hash-backend indexes for two chunk sizes, and a template test set."""
    from drug_rag.embeddings import make_embedder
    from drug_rag.index import build_index
    from drug_rag.testset import build_testset

    (tmp_path / "data").mkdir()
    with open(tmp_path / "data" / "labels.jsonl", "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    emb = make_embedder("hash")
    for cs in (300, 600):
        build_index(labels, cs, emb, root=tmp_path / "data" / "index", progress=False)
    qs = build_testset(labels, None, 20, 0, tmp_path / "data" / "testset.jsonl")
    return tmp_path, qs
