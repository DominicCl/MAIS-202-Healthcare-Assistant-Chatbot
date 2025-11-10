import torch
from chatbot import chatbot

# Load your disease list
diseases = [
    "Acne",
    "Alzheimer's Disease",
    "Anemia",
    "Appendicitis",
    "Arthritis",
    "Asthma",
    "Atherosclerosis",
    "Autism Spectrum Disorder (ASD)",
    "Bipolar Disorder",
    "Bronchitis",
    "Cancer (General)",
    "Cataracts",
    "Celiac Disease",
    "Chickenpox (Varicella)",
    "Chronic Obstructive Pulmonary Disease (COPD)",
    "Cirrhosis (Liver)",
    "Common Cold",
    "Congestive Heart Failure",
    "COVID-19",
    "Crohn's Disease",
    "Cystic Fibrosis (CF)",
    "Dementia",
    "Depression (Major)",
    "Diabetes Mellitus (Type 2)",
    "Ebola Virus Disease",
    "Endometriosis",
    "Epilepsy",
    "Fibromyalgia",
    "Gout",
    "Hepatitis B",
    "High Blood Pressure (Hypertension)",
    "HIV/AIDS",
    "Influenza (Flu)",
    "Irritable Bowel Syndrome (IBS)",
    "Kidney Stones",
    "Lyme Disease",
    "Measles",
    "Multiple Sclerosis (MS)",
    "Parkinson's Disease",
    "Schizophrenia",
    "Stroke (Cerebrovascular Accident)"
]

# Initialize chatbot
bot = chatbot(
    disease_model_path="disease_predictor_final.pt",
    disease_list=diseases,
    device=torch.device("cpu")  # or "cuda" if you have a GPU
)

# Interactive loop
print("Chatbot ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = bot.chat(user_input)
    print("Bot:", response)
