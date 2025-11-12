import torch
from dataset import MultiLabelTextDataset
from model import MultiLabelClassificationModel

CSV_VAL = "../data/val.csv"
MODEL_PATH = "disease_predictor_final.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_features(symptoms, vocab):
    x = torch.zeros(len(vocab))
    for s in symptoms:
        if s in vocab:
            x[vocab.index(s)] = 1.0
    return x

# load dataset
ds = MultiLabelTextDataset(CSV_VAL)
vocab = ds.symptom_vocab
labels = ds.label_list

model = MultiLabelClassificationModel(
    "MLC-FastTrainer",
    len(labels),
    len(vocab),
    dropout=0.10
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def run_case(name, symptoms):
    x = make_features(symptoms, vocab).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    top = probs.argsort()[::-1][:5]

    print(f"\n Case: {name}")
    print("Symptoms:", ", ".join(symptoms))
    print("Predictions:")
    for i in top:
        print(f"  - {labels[i]:30s} {probs[i]*100:.2f}%")
    print("-"*60)

# test cases
cases = [
    ("Flu-like illness", ["fever", "headache", "fatigue", "sore_throat", "runny_nose", "body_pain"]),
    ("Digestive upset", ["abdominal_pain", "vomiting", "diarrhoea", "nausea", "dehydration"]),
    ("Skin rash and itching", ["itching", "skin_rash", "red_spots_over_body", "fatigue"]),
    ("Severe chest pain", ["chest_pain", "shortness_of_breath", "fatigue", "sweating"]),
    ("Allergy reaction", ["sneezing", "runny_nose", "itchy_eyes", "nasal_congestion"]),
    ("Migraine episode", ["headache", "nausea", "sensitivity_to_light", "dizziness"]),
    ("Thyroid problem", ["fatigue", "weight_gain", "constipation", "dry_skin", "hair_loss"]),
    ("UTI", ["burning_sensation", "frequent_urination", "lower_abdominal_pain", "back_pain", "fever"]),
    ("Jaundice-like illness", ["yellowing_of_eyes", "fatigue", "nausea", "loss_of_appetite"]),
    ("Asthma attack", ["cough", "wheezing", "shortness_of_breath", "chest_tightness"])
]

for name, syms in cases:
    run_case(name, syms)
