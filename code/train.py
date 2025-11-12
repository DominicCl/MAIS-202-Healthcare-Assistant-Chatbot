import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import pandas as pd

from dataset import MultiLabelTextDataset
from model import MultiLabelClassificationModel
from metrics import MetricsEvaluator


# CONFIG

CSV_TRAIN = "../data/train.csv"
CSV_VAL = "../data/val.csv"
MODEL_NAME = "MLC-FastTrainer"
BATCH_SIZE = 64
LR = 0.0025
EPOCHS = 100
DROPOUT = 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.40  # threshold for multilabel classification



# DATA LOADER HELPER

def make_loader(csv_path, label_list=None, symptom_vocab=None, batch_size=64, shuffle=False):
    ds = MultiLabelTextDataset(csv_path, tokenizer_name=MODEL_NAME, label_list=label_list)

    # Force consistent symptom vocab between train and val
    if symptom_vocab is not None:
        ds.symptom_vocab = symptom_vocab
        ds.num_features = len(symptom_vocab)
        ds.symptom_to_index = {s: i for i, s in enumerate(symptom_vocab)}

    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return ds, dl



# MAIN TRAINING LOOP

def main():
    t0 = time.time()

    # Load training data (defines labels + vocab)
    train_ds, train_loader = make_loader(CSV_TRAIN, label_list=None, batch_size=BATCH_SIZE, shuffle=True)

    # Load validation data (same label list + same vocab)
    val_ds, val_loader = make_loader(
        CSV_VAL,
        label_list=train_ds.label_list,
        symptom_vocab=train_ds.symptom_vocab,
        batch_size=32,
        shuffle=False,
    )

    num_labels = len(train_ds.label_list)
    num_features = train_ds.num_features

    print(f"\n Diseases: {num_labels} |  Features: {num_features}")

    #  Model
    model = MultiLabelClassificationModel(
        encoder_name=MODEL_NAME,
        num_labels=num_labels,
        input_features=num_features,
        dropout=DROPOUT
    ).to(DEVICE)

    # Class weights (inverse frequency)
    counts = pd.read_csv(CSV_TRAIN)["Disease"].value_counts()
    class_weights = [1.0 / (counts.get(lbl, 1) / counts.sum()) for lbl in train_ds.label_list]
    pos_weight = torch.tensor(class_weights, dtype=torch.float, device=DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = Adam(model.parameters(), lr=LR)
    evaluator = MetricsEvaluator(DEVICE)

    print("\n Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            x = batch["features"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

            if step == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))

        #  Evaluate on validation set
        val_acc, val_prec, val_rec, val_f1 = evaluator.evaluate(
            model, val_loader, threshold=THRESHOLD, debug=(epoch == 0)
        )
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | "
              f"Val_Acc: {val_acc:.4f} | Val_Prec: {val_prec:.4f} | "
              f"Val_Rec: {val_rec:.4f} | Val_F1: {val_f1:.4f}")

      

    #  Save final model
    torch.save(model.state_dict(), "disease_predictor_final.pt")
    print(f"\n Done in {(time.time() - t0)/60:.2f} min | Saved 'disease_predictor_final.pt'.")
    patience = 10
    best_f1 = 0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        train_loss = train(...)
        val_acc, val_prec, val_rec, val_f1 = evaluate(...)
    
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1
    
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping at epoch {epoch}")
            break




if __name__ == "__main__":
    main()

