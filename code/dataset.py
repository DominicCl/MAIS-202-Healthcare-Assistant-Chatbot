import torch
from torch.utils.data import Dataset
import pandas as pd

class MultiLabelTextDataset(Dataset):
    def __init__(self, dataset_csv_path, tokenizer_name=None, label_list=None):
        super().__init__()
        self.data = pd.read_csv(dataset_csv_path)

        # Clean columns and disease strings
        self.data.columns = [col.strip() for col in self.data.columns]
        self.data["Disease"] = (
            self.data["Disease"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
        )

        # Build label list (diseases)
        if label_list is None:
            self.label_list = sorted(self.data["Disease"].unique())
        else:
            self.label_list = label_list
        self.num_labels = len(self.label_list)
        self.label_to_index = {label: i for i, label in enumerate(self.label_list)}

        # Build symptom vocab
        symptom_cols = [c for c in self.data.columns if c.lower() != "disease"]
        raw = self.data[symptom_cols].values.ravel()
        unique_syms = [s for s in raw if isinstance(s, str) and s.strip() != ""]
        self.symptom_vocab = sorted(set(s.strip() for s in unique_syms))
        self.num_features = len(self.symptom_vocab)
        self.symptom_to_index = {s: i for i, s in enumerate(self.symptom_vocab)}

        print(f"Loaded {len(self.data)} samples")
        print(f"Detected {self.num_features} unique symptoms")
        print(f"Detected {self.num_labels} unique diseases")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        disease_name = row["Disease"]

        # Build features
        features = torch.zeros(self.num_features, dtype=torch.float)
        for col in self.data.columns:
            if col.lower() == "disease":
                continue
            val = row[col]
            if isinstance(val, str):
                val = val.strip()
            if isinstance(val, str) and val in self.symptom_to_index:
                features[self.symptom_to_index[val]] = 1.0

        # Build labels (one-hot)
        labels = torch.zeros(self.num_labels, dtype=torch.float)
        if disease_name in self.label_to_index:
            labels[self.label_to_index[disease_name]] = 1.0

        return {"features": features, "labels": labels}
