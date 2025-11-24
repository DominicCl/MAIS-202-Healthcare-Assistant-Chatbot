import torch
import numpy as np
from model import SymptomNetV2

def load_nn_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    model = SymptomNetV2(checkpoint["num_features"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["classes"], checkpoint["symptom_cols"]


def predict_from_vector(x, model, classes, top_k=3):
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).numpy().ravel()

    top_idx = probs.argsort()[::-1][:top_k]
    return [(classes[i], float(probs[i])) for i in top_idx]
