# app.py
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

from symptom_extractor import extract_symptoms  


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "code" / "best_nn_model.pt"


# -------------------------------------------------
# Model architecture that match training
# -------------------------------------------------
class SymptomNetV2(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.3):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
def load_nn_model():
   
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model_state = bundle["model_state_dict"]
    num_features = bundle["num_features"]
    num_classes = bundle["num_classes"]
    symptom_cols = bundle["symptom_cols"]
    label_classes = bundle["label_classes"]

    # Build Nural net  architecture
    model = SymptomNetV2(
        num_features=num_features,
        num_classes=num_classes,
        dropout=0.3
    )
    model.load_state_dict(model_state)
    model.eval()

    return model, label_classes, symptom_cols


# Load model 
MODEL, LABEL_CLASSES, SYMPTOM_COLS = load_nn_model()


# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__)
CORS(app)


# -------------------------------------------------
# api chat rout
# -------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    text = data.get("message", "")

    # 1. Extract symptoms
    symptoms = extract_symptoms(text)

    # 2. Convert to NN vector
    x = np.array([1 if s in symptoms else 0 for s in SYMPTOM_COLS], dtype=np.float32)
    x_tensor = torch.tensor([x], dtype=torch.float32)

    # 3. Run NN
    with torch.no_grad():
        logits = MODEL(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    # 4. Top 3 predictions
    top_idx = np.argsort(probs)[::-1][:3]
    results = [
        {"disease": LABEL_CLASSES[i], "probability": float(probs[i])}
        for i in top_idx
    ]

    return jsonify({
        "symptoms": symptoms,
        "predictions": results
    })


# -------------------------------------------------
# reset api
# -------------------------------------------------
@app.post("/api/reset")
def api_reset():
    return jsonify({"status": "reset"})


# -------------------------------------------------
#running 
# -------------------------------------------------
@app.get("/")
def index():
    return jsonify({
        "message": "Medical Diagnosis AI API running.",
        "endpoints": ["/api/chat", "/api/reset"]
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
