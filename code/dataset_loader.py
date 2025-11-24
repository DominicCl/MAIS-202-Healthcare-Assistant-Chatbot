# dataset_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    df["disease"] = df["disease"].str.strip()

    # All columns except disease are symptoms
    symptom_cols = [c for c in df.columns if c != "disease"]

    X = df[symptom_cols].astype("float32")
    y_strings = df["disease"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_strings)

    num_classes = len(label_encoder.classes_)
    print(f"Detected {len(symptom_cols)} symptoms and {num_classes} diseases.")

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder,   
        symptom_cols
    )
