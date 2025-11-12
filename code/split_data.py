import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "../data/dataset_augmented.csv"
TRAIN_OUT = "../data/train.csv"
VAL_OUT = "../data/val.csv"

if "train" in SRC.lower() or "val" in SRC.lower():
    raise ValueError(" You're trying to split a split file Use dataset.csv or dataset_augmented.csv instead.")

def normalize_disease_name(s):
    if pd.isna(s):
        return ""
    return (
        str(s).strip().lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", "")
        .replace(",", "")
        .replace("  ", " ")
    )

def main():
    df = pd.read_csv(SRC)
    print(f"Original dataset size: {len(df)}")

    # Clean columns & disease names
    df.columns = [c.strip() for c in df.columns]
    df["Disease"] = df["Disease"].apply(normalize_disease_name)

    # Remove exact duplicates
    df = df.drop_duplicates()
    print(f"After removing exact duplicates: {len(df)}")

    # Remove symptom-level duplicates (identical symptom sets)
    symptom_cols = [c for c in df.columns if c.lower() != "disease"]
    df["_sig"] = df[symptom_cols].astype(str).agg("-".join, axis=1)
    df = df.drop_duplicates(subset=["_sig", "Disease"]).drop(columns=["_sig"])
    print(f"After removing symptom-level duplicates: {len(df)}")

    # Try stratified split
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=0.1,
            stratify=df["Disease"],
            random_state=42,
            shuffle=True,
        )
        print(" Stratified split complete!")
    except ValueError as e:
        print(" Stratified split failed; falling back to random split.")
        print("   Details:", e)
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42, shuffle=True
        )

    # Verify overlap
    overlap = pd.merge(train_df, val_df, how="inner", on=list(df.columns))
    print(f"Overlap rows: {len(overlap)}")
    if len(overlap) == 0:
        print(" Great! No overlap — clean split achieved.")
    else:
        print(" Overlap exists — check for unclean text fields.")

    # Save
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

if __name__ == "__main__":
    main()
