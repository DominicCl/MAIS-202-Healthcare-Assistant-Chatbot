import pandas as pd
import numpy as np
from collections import Counter

SRC = "../data/dataset.csv"   # use the original as base
OUT = "../data/dataset_augmented.csv"

# Parameters
TARGET_SAMPLES_PER_DISEASE = 300
MIN_SYMPTOMS = 3
MAX_SYMPTOMS = 10
NOISE_RATE = 0.1
SEED = 42


def normalize_disease_name(s):
    """Standardize disease names (lowercase, remove punctuation)"""
    if pd.isna(s):
        return ""
    return (
        str(s)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", "")
        .replace(",", "")
    )


def main():
    np.random.seed(SEED)

    # Load dataset
    df = pd.read_csv(SRC)
    df.columns = [c.strip() for c in df.columns]
    df["Disease"] = df["Disease"].apply(normalize_disease_name)

    # Detect symptom columns
    symptom_cols = [c for c in df.columns if c.lower() != "disease"]
    print(f"Detected symptom columns: {symptom_cols}")
    print(f"Loaded {len(df)} rows from {SRC}")

    diseases = sorted(df["Disease"].unique())
    print(f"Found {len(diseases)} diseases")

    # Build disease → symptom frequency map
    disease_symptom_freq = {}
    all_symptoms = set()

    for dis in diseases:
        subset = df[df["Disease"] == dis]
        sym_counts = Counter()

        for _, row in subset.iterrows():
            for col in symptom_cols:
                val = str(row[col]).strip()
                if val and val.lower() != "nan":
                    sym_counts[val] += 1
                    all_symptoms.add(val)

        total = sum(sym_counts.values())
        if total == 0:
            continue

        freq = {s: sym_counts[s] / total for s in sym_counts}
        disease_symptom_freq[dis] = freq

    all_symptoms = sorted(all_symptoms)
    print(f"Total unique symptoms: {len(all_symptoms)}")

    # Synthetic data generation 
    synthetic_rows = []

    for disease, freq_map in disease_symptom_freq.items():
        symptoms = list(freq_map.keys())
        probs = np.array(list(freq_map.values()))
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)

        existing = len(df[df["Disease"] == disease])
        n_to_gen = max(0, TARGET_SAMPLES_PER_DISEASE - existing)
        print(f"→ {disease}: generating {n_to_gen} synthetic samples")

        for _ in range(n_to_gen):
            n_sym = np.random.randint(MIN_SYMPTOMS, min(len(symptoms), MAX_SYMPTOMS) + 1)
            chosen = np.random.choice(symptoms, size=n_sym, replace=False, p=probs).tolist()

            # Add small noise (random extra symptom)
            if np.random.rand() < NOISE_RATE:
                noise_sym = np.random.choice(all_symptoms)
                if noise_sym not in chosen:
                    chosen.append(noise_sym)

            row = {col: "" for col in symptom_cols}
            for i, sym in enumerate(chosen[:len(symptom_cols)]):
                row[symptom_cols[i]] = sym

            row["Disease"] = disease
            synthetic_rows.append(row)

    # Combine real + synthetic data
    synth_df = pd.DataFrame(synthetic_rows)
    full_df = pd.concat([df, synth_df], ignore_index=True)

    print(f"\n Generated {len(synthetic_rows)} synthetic samples")
    print(f" Total dataset size after augmentation: {len(full_df)}")

    # Save
    full_df.to_csv(OUT, index=False)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

