import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "synthetic_medical_dataset.csv"
OUT_PATH = BASE_DIR.parent / "data" / "symptom_columns.json"

print("Loading dataset:", CSV_PATH)

df = pd.read_csv(CSV_PATH)

# All columns except the disease label
symptom_cols = [c for c in df.columns if c != "disease"]

print(f"Detected {len(symptom_cols)} symptom columns.")

with open(OUT_PATH, "w") as f:
    json.dump(symptom_cols, f, indent=4)

print("Saved:", OUT_PATH)
