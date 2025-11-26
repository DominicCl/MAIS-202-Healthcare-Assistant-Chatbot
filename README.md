# MAIS-202-Healthcare-Assistant-Chatbot
 AI Medical Diagnosis Assistant

Symptom to a  Neural Network Disease Prediction Web App

This project is a complete AI-driven medical assistant that:
- Extracts symptoms from plain English text
- Converts them into a structured vector
- Feeds them into a trained Neural Network
- Returns the Topn3 most likely diseases with probabilities
- Runs in a clean web interface

This project is for educational use and should not  be used for real medical decisions.
📂 Project Structure
AImodel/
│── code/
│    ├── app.py                # Flask backend API
│    ├── train.py              # Neural network training pipeline
│    ├── dataset_loader.py     # Data loading + splits
│    ├── symptom_extractor.py  # NLP extractor (semantic + fuzzy)
│    ├── model.py              # SymptomNetV2 definition
│    ├── best_nn_model.pt      # Trained model checkpoint
│
│── data/
│    ├── symptom_columns.json  # Full symptom vocabulary (157 symptoms)
│    ├── synthetic_dataset.csv # 35,000 realistic patient profiles
│
│── website/
│    ├── index.html            # Frontend UI
│    ├── styles.css
│    ├── script.js
│
│── requirements.txt
│── README.md 

Quick Start (Run the Full Web App Locally)
1. Install dependencies

Make sure you are in the project folder:

pip install -r requirements.txt

2. Run the backend API

From inside the code/ directory:

python app.py


3. Open the Frontend
Option A — Open locally through file system

Just open:

website/index.html
in a browser

API Endpoints
POST /api/chat

Send a user symptom description.

Request body:

{ "message": "I have chest pain and shortness of breath" }


Response:

{
  "symptoms": ["chest_pain", "shortness_of_breath"],
  "predictions": [
    { "disease": "myocardial_infarction", "probability": 0.94 },
    { "disease": "stable_angina", "probability": 0.03 },
    { "disease": "aortic_dissection", "probability": 0.01 }
  ]
}

Neural Network Model

The model is a deep feed-forward neural network:

157 input symptoms →
512 → 256 → 128 →
61-class output (softmax)


Hyperparameters discovered viagrid search:

Optimizer: Adam

Dropout: 0.3

Early stopping

Weight decay: 1e-4

Learning rate: 1e-3

Training

If you want to retrain the neural network from scratch:

cd code/
python train.py


The script:

loads synthetic_dataset.csv

extracts symptom columns

splits into train / val / test

auto-tunes multiple architectures

saves the best model to:

data/best_nn_model.pt

Symptom Extraction Engine

The extractor uses:

- Sentence-BERT (MiniLM-L6-v2) semantic matching
- Fuzzy matching (difflib)
- Literal matching
- Custom synonyms for natural language

Example:

"I'm puking with stomach cramps and can't breathe well"

Extracts:

["vomiting", "cramps", "shortness_of_breath"]
