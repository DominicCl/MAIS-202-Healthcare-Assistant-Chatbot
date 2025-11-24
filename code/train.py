
#Full Neural Network Training Pipeline
# Works with: dataset_loader.py (returns 8 values)

import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_loader import load_dataset



# 1. Model Definition work with 1–3 hidden layers
class SymptomNetV2(nn.Module):
    def __init__(self, num_features, num_classes,
                 hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)



# 2. Train One Model
def train_one_model(X_train, y_train, X_val, y_val,
                    num_features, num_classes,
                    hidden1, hidden2,
                    lr, dropout, weight_decay,
                    device):

    model = SymptomNetV2(
        num_features=num_features,
        num_classes=num_classes,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Convert to TensorLoader
    train_ds = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    # Early stopping parameters cuz if not takes too long
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                val_losses.append(loss.item())

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        val_loss = np.mean(val_losses)
        val_acc = correct / total

        print(f"Epoch {epoch+1:02d}: "
              f"Train Loss={np.mean(train_losses):.4f} | "
              f"Val Loss={val_loss:.4f} | "
              f"Val Acc={val_acc:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # return best model
    return best_model_state, val_acc, best_val_loss



# 3. search for best Hyperparameter 
def hyperparameter_search(csv_path, output_path, max_seconds=1800):
    start_time = time.time()

    print("Loading dataset")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
        symptom_cols
    ) = load_dataset(csv_path)

    num_features = len(symptom_cols)
    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_features} symptoms and {num_classes} diseases.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # hyperparameter search space
    hidden_sizes = [(512, 256), (256, 128), (128, 64)]
    lrs = [1e-3, 5e-4]
    dropouts = [0.2, 0.3]
    weight_decays = [1e-4, 1e-5]

    best_overall = None

    for h1, h2 in hidden_sizes:
        for lr in lrs:
            for dp in dropouts:
                for wd in weight_decays:

                    if time.time() - start_time > max_seconds:
                        print("⏳ Time limit reached, stopping search.")
                        break

                    print("\n====================================")
                    print(f"Testing config: h1={h1}, h2={h2}, lr={lr}, dropout={dp}, wd={wd}")
                    print("====================================")

                    model_state, val_acc, val_loss = train_one_model(
                        X_train, y_train,
                        X_val, y_val,
                        num_features, num_classes,
                        hidden1=h1,
                        hidden2=h2,
                        lr=lr,
                        dropout=dp,
                        weight_decay=wd,
                        device=device,
                    )

                    if (best_overall is None) or (val_acc > best_overall["acc"]):
                        best_overall = {
                            "state": model_state,
                            "acc": val_acc,
                            "loss": val_loss,
                            "params": (h1, h2, lr, dp, wd),
                        }
                        print(f" New best model! Val Acc={val_acc:.4f}")

    print("\n BEST MODEL FOUND")
    print(best_overall)

    # Save final model bundle
    torch.save({
        "model_state_dict": best_overall["state"],
        "num_features": num_features,
        "num_classes": num_classes,
        "symptom_cols": symptom_cols,
        "label_classes": label_encoder.classes_
    }, output_path)

    print(f"\nSaved best model to {output_path}")



# 4. MAIN ENTRY POINT
if __name__ == "__main__":
    csv_path = "../data/synthetic_medical_dataset.csv"
    output_path = "../data/best_nn_model.pt"

    hyperparameter_search(
        csv_path=csv_path,
        output_path=output_path,
        max_seconds=1800  # 30-minute limit
    )
