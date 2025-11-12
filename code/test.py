import torch, random
from dataset import MultiLabelTextDataset
from model import MultiLabelClassificationModel

MODEL_PATH = "disease_predictor_final.pt"
CSV_VAL = "../data/val.csv"
MODEL_NAME = "MLC-FastTrainer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.40

dataset = MultiLabelTextDataset(CSV_VAL, MODEL_NAME)
num_labels = len(dataset.label_list)
num_features = dataset.num_features

model = MultiLabelClassificationModel(
    encoder_name=MODEL_NAME,
    num_labels=num_labels,
    input_features=num_features,
    dropout=0.10
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f" Loaded {MODEL_PATH}")
print(f"{len(dataset)} samples | {num_features} symptoms | {num_labels} diseases")

idx = random.randint(0, len(dataset)-1)
sample = dataset[idx]
x = sample["features"].unsqueeze(0).to(DEVICE)
true_idx = torch.argmax(sample["labels"]).item()
true_name = dataset.label_list[true_idx]

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy().ravel()

top = probs.argsort()[::-1][:5]
print("\nRandom Sample Test")
print(f"True disease: {true_name}")
for i in top:
    print(f" - {dataset.label_list[i]} ({probs[i]*100:.2f}%)")

# Validation accuracy
correct = 0
for s in dataset:
    x = s["features"].unsqueeze(0).to(DEVICE)
    y = torch.argmax(s["labels"]).item()
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    if pred == y:
        correct += 1
print(f"\n Val Accuracy: {correct/len(dataset):.4f}")

# Challenge test
active = [i for i, v in enumerate(sample["features"]) if v == 1.0]
if active:
    removed = random.choice(active)
    x2 = sample["features"].clone()
    x2[removed] = 0.0
    with torch.no_grad():
        logits2 = model(x2.unsqueeze(0).to(DEVICE))
        probs2 = torch.sigmoid(logits2).cpu().numpy().ravel()
    top2 = probs2.argsort()[::-1][:5]
    print(f"\n Removed symptom: {dataset.symptom_vocab[removed]}")
    for i in top2:
        print(f" - {dataset.label_list[i]} ({probs2[i]*100:.2f}%)")
