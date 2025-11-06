import torch 
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import List, Dict, Tuple
from model import MultiLabelClassificationModel

# Combining chatbot wrapper with disease predictor
class chatbot: 
  # initializing the chatbot by loading in the conversational model
  def __init__(self, 
              disease_model_path: str, # instantiates our model
              disease_list: List[str], # passing our list of disease 
              device: torch.device, # convention
              conversational_model_name: str = "microsoft/DialoGPT-small", # chatbot
              explain_ml: bool = True): # flag
    
    self.device = device
    self.disease_list = disease_list
    self.num_diseases = len(disease_list)
    self.explain_ml = explain_ml

    # loading our mutlilabel classification model
    print("Loading the Multi Label Classification Model")
    self.disease_model = MultiLabelClassificationModel(
      encoder_name='MLC-Logistic-Regression-Model',
      num_labels=self.num_diseases,
      dropout=0.1
    ).to(device)

    # this loads the best trained weights into the actual model, so we can use it at its best
    self.disease_model.load_state_dict(torch.load(disease_model_path, map_location=device))
    self.disease_model.eval() # this sets the model to evaluation mode so we dont use dropout during inference, essentially switching it from training mode to prediction mode (ofc we dont want dropout during predictions)

    # NOW THE CHATBOT WRAPPER
    self.tokenizer = AutoTokenizer.from_pretrained(conversational_model_name)
    self.conversational_model = AutoModel.from_pretrained (conversational_model_name).to(device)

    # mfix the issue of unequal sequence lengths by giving the tokenizer a filler/pad to make all sentences in a batch the same size
    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.conversational_model.resize_token_embeddings(len(self.tokenizer)) # telling the model about the resize on the tokens so it can adjust its lookup table to match it

    # mapping the numbers the model understands to the actual diseases
    self.symptom_keywords = self._initialize_symptom_mapping()
    
    # holding the conversation statte
    self.conversation_history = []
    self.collected_symptoms = []
    self.ml_concepts_explained = []

  def _initialize_symptom_mapping(self) -> Dict: # mapping num to actual disease
      return {
          'fever': 1,
          'cough': 2,
          'fatigue': 3,
          'tired': 3,
          'exhausted': 3,
          'headache': 4,
          'nausea': 5,
          'vomiting': 5,
          'pain': 6,
          'ache': 6,
          'breathing': 7,
          'breath': 7,
          'shortness of breath': 7,
          'dizziness': 8,
          'dizzy': 8,
          'lightheaded': 8,
          'rash': 9,
          'skin': 9,
      }
    
  def extract_symptoms_from_text(self, user_input: str) -> List[str]:
    # this will extract the symptoms the user inputted in their text to use them to guess the diseases
    user_input_lower = user_input.lower() # all of str to lowercase
    detected_symptoms = []
    
    for symptom in self.symptom_keywords.keys():
        if symptom in user_input_lower:
            detected_symptoms.append(symptom)
    
    return detected_symptoms
    
  def symptoms_to_features(self, symptoms: List[str]) -> torch.Tensor:
      # map symptom to feature so our model can guess
      features = torch.zeros(10, dtype=torch.float)

      for symptom in symptoms:
        if symptom in self.symptom_keywords:
            feature_index = self.symptom_keywords[symptom]
            features[feature_index] = 1.0
      
      features[1:] += 0.01 * torch.randn(9) # adding some noise because our model was trained with it
      return features
  
  def predict_disease(self, features: torch.Tensor) -> Tuple[List[str], List[float], torch.Tensor]:
      # preparing the input tensor for the model
      features = features.unsqueeze(0).to(self.device)
      with torch.no_grad(): #d disable gradient calculation during prediction
        logits = self.disease_model(features) # what the model returns

      # actually convert the logits into usable probability
      probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
      # getting the top 3 predictions
      top_indices = np.argsort(probabilities)[-3:][::-1]
      top_diseases = [self.disease_list[idx] for idx in top_indices]
      top_probs = [probabilities[idx] for idx in top_indices]
      
      return top_diseases, top_probs, logits
    
  # MISSING METHOD 1
  def explain_prediction(self, symptoms: List[str], diseases: List[str], probs: List[float]) -> str:
      explanation = ""
      
      if self.explain_ml and 'mlp_forward_pass' not in self.ml_concepts_explained:
          explanation += "\n🎓 **ML Concept - MLP Forward Pass**\n"
          explanation += "Symptoms → 10D vector → Hidden(128, ReLU) → Output(41, Sigmoid) → Probabilities\n\n"
          self.ml_concepts_explained.append('mlp_forward_pass')
      
      explanation += f"📊 **Predictions:**\n"
      for i, (disease, prob) in enumerate(zip(diseases, probs), 1):
          bar = "█" * int(prob * 20)
          explanation += f"{i}. {disease}: {prob*100:.1f}% {bar}\n"
      
      return explanation
    
  # MISSING METHOD 2
  def compare_with_other_models(self) -> str:
      return """
      📚 **Comparing ML Models (MAIS Lectures)**

      **1. KNN (Lecture 1):** Find K nearest patients, predict majority disease
      **2. Decision Tree (Lecture 2):** Rules like "if fever AND cough then Disease_X"
      **3. Logistic Regression (Lecture 3):** Linear model with sigmoid
      **4. MLP - Your Model! (Lecture 4):** Non-linear with hidden layers
      **5. CNN (Lecture 5):** For images, not applicable here
      **6. RNN/LSTM (Lecture 6):** For sequences, could track symptom progression
      """
    
  # MISSING METHOD 3
  def _explain_architecture(self) -> str:
      return """
      🏗️ **MLP Architecture**

      Input(10) → Hidden(128, ReLU) → Dropout(0.01) → Output(41, Sigmoid)

      **Training:** Binary Cross-Entropy loss, Adam optimizer, 300 epochs
      **Forward:** h = ReLU(W₁·x + b₁), ŷ = σ(W₂·h + b₂)
      """
    
  def generate_response(self, user_input: str) -> str:
      if "explain model" in user_input.lower() or "compare models" in user_input.lower():
          return self.compare_with_other_models()
      
      if "how does it work" in user_input.lower():
          return self._explain_architecture()
      
      new_symptoms = self.extract_symptoms_from_text(user_input)
      self.collected_symptoms.extend(new_symptoms)
      self.collected_symptoms = list(set(self.collected_symptoms))
      
      if len(self.collected_symptoms) >= 1:
          features = self.symptoms_to_features(self.collected_symptoms)
          diseases, probs, logits = self.predict_disease(features)
          
          response = f"Based on symptoms: **{', '.join(self.collected_symptoms)}**\n\n"
          response += self.explain_prediction(self.collected_symptoms, diseases, probs)
          response += "\n\n💬 Ask 'compare models' or 'how does it work'\n"
          
      elif "hello" in user_input.lower() or "hi" in user_input.lower():
          response = "👋 Hello! I'm an ML-powered medical assistant!\n\n"
          response += "What symptoms are you experiencing?\n"
          response += "(e.g., fever, cough, headache, fatigue, nausea)\n"
      else:
          response = "Please describe your symptoms (e.g., 'I have fever and cough')\n"
      
      return response
  
  # MISSING METHOD 4
  def chat(self, user_input: str) -> str:
      self.conversation_history.append({"role": "user", "content": user_input})
      response = self.generate_response(user_input)
      self.conversation_history.append({"role": "assistant", "content": response})
      return response
    
  # MISSING METHOD 5
  def reset_conversation(self):
      self.conversation_history = []
      self.collected_symptoms = []
      self.ml_concepts_explained = []
    
  # MISSING METHOD 6
  def save_conversation(self, filepath: str):
      with open(filepath, 'w') as f:
          json.dump(self.conversation_history, f, indent=2)


import torch 
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from typing import List, Dict, Tuple
from model import MultiLabelClassificationModel

# Combining chatbot wrapper with disease predictor
class chatbot: 
  # initializing the chatbot by loading in the conversational model
  def __init__(self, 
              disease_model_path: str, # instantiates our model
              disease_list: List[str], # passing our list of disease 
              device: torch.device, # convention
              conversational_model_name: str = "microsoft/DialoGPT-small", # chatbot
              explain_ml: bool = True): # flag
    
    self.device = device
    self.disease_list = disease_list
    self.num_diseases = len(disease_list)
    self.explain_ml = explain_ml

    # loading our mutlilabel classification model
    print("Loading the Multi Label Classification Model")
    self.disease_model = MultiLabelClassificationModel(
      encoder_name='MLC-Logistic-Regression-Model',
      num_labels=self.num_diseases,
      dropout=0.1
    ).to(device)

    # this loads the best trained weights into the actual model, so we can use it at its best
    self.disease_model.load_state_dict(torch.load(disease_model_path, map_location=device))
    self.disease_model.eval() # this sets the model to evaluation mode so we dont use dropout during inference, essentially switching it from training mode to prediction mode (ofc we dont want dropout during predictions)

    # NOW THE CHATBOT WRAPPER
    self.tokenizer = AutoTokenizer.from_pretrained(conversational_model_name)
    self.conversational_model = AutoModel.from_pretrained (conversational_model_name).to(device)

    # mfix the issue of unequal sequence lengths by giving the tokenizer a filler/pad to make all sentences in a batch the same size
    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.conversational_model.resize_token_embeddings(len(self.tokenizer)) # telling the model about the resize on the tokens so it can adjust its lookup table to match it

    # mapping the numbers the model understands to the actual diseases
    self.symptom_keywords = self._initialize_symptom_mapping()
    
    # holding the conversation statte
    self.conversation_history = []
    self.collected_symptoms = []
    self.ml_concepts_explained = []

  def _initialize_symptom_mapping(self) -> Dict: # mapping num to actual disease
      return {
          'fever': 1,
          'cough': 2,
          'fatigue': 3,
          'tired': 3,
          'exhausted': 3,
          'headache': 4,
          'nausea': 5,
          'vomiting': 5,
          'pain': 6,
          'ache': 6,
          'breathing': 7,
          'breath': 7,
          'shortness of breath': 7,
          'dizziness': 8,
          'dizzy': 8,
          'lightheaded': 8,
          'rash': 9,
          'skin': 9,
      }
  
  def extract_symptoms_from_text(self, user_input: str) -> List[str]:
    # this will extract the symptoms the user inputted in their text to use them to guess the diseases
    user_input_lower = user_input.lower() # all of str to lowercase
    detected_symptoms = []
    
    for symptom in self.symptom_keywords.keys():
        if symptom in user_input_lower:
            detected_symptoms.append(symptom)
    
    return detected_symptoms
  
  def symptoms_to_features(self, symptoms: List[str]) -> torch.Tensor:
      # map symptom to feature so our model can guess
      features = torch.zeros(10, dtype=torch.float)

      for symptom in symptoms:
        if symptom in self.symptom_keywords:
            feature_index = self.symptom_keywords[symptom]
            features[feature_index] = 1.0
      
      features[1:] += 0.01 * torch.randn(9) # adding some noise because our model was trained with it
      return features
  
  def predict_disease(self, features: torch.Tensor) -> Tuple[List[str], List[float], torch.Tensor]:
      # preparing the input tensor for the model
      features = features.unsqueeze(0).to(self.device)
      with torch.no_grad(): #d disable gradient calculation during prediction
        logits = self.disease_model(features) # what the model returns

      # actually convert the logits into usable probability
      probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
      # getting the top 3 predictions
      top_indices = np.argsort(probabilities)[-3:][::-1]
      top_diseases = [self.disease_list[idx] for idx in top_indices]
      top_probs = [probabilities[idx] for idx in top_indices]
      
      return top_diseases, top_probs, logits
  
  def explain_prediction(self, symptoms: List[str], diseases: List[str], probs: List[float]) -> str:
      explanation = ""
      
      if self.explain_ml and 'mlp_forward_pass' not in self.ml_concepts_explained:
          explanation += "\n🎓 **ML Concept - MLP Forward Pass**\n"
          explanation += "Symptoms → 10D vector → Hidden(128, ReLU) → Output(41, Sigmoid) → Probabilities\n\n"
          self.ml_concepts_explained.append('mlp_forward_pass')
      
      explanation += f"📊 **Predictions:**\n"
      for i, (disease, prob) in enumerate(zip(diseases, probs), 1):
          bar = "█" * int(prob * 20)
          explanation += f"{i}. {disease}: {prob*100:.1f}% {bar}\n"
      
      return explanation
  
  
  def generate_response(self, user_input: str) -> str:
      if "explain model" in user_input.lower() or "compare models" in user_input.lower():
          return self.compare_with_other_models()
      
      if "how does it work" in user_input.lower():
          return self._explain_architecture()
      
      new_symptoms = self.extract_symptoms_from_text(user_input)
      self.collected_symptoms.extend(new_symptoms)
      self.collected_symptoms = list(set(self.collected_symptoms))
      
      if len(self.collected_symptoms) >= 1:
          features = self.symptoms_to_features(self.collected_symptoms)
          diseases, probs, logits = self.predict_disease(features)
          
          response = f"Based on symptoms: **{', '.join(self.collected_symptoms)}**\n\n"
          response += self.explain_prediction(self.collected_symptoms, diseases, probs)
          response += "\n\n💬 Ask 'compare models' or 'how does it work'\n"
          
      elif "hello" in user_input.lower() or "hi" in user_input.lower():
          response = "👋 Hello! I'm an ML-powered medical assistant!\n\n"
          response += "What symptoms are you experiencing?\n"
          response += "(e.g., fever, cough, headache, fatigue, nausea)\n"
      else:
          response = "Please describe your symptoms (e.g., 'I have fever and cough')\n"
      
      return response
  
  def chat(self, user_input: str) -> str:
      self.conversation_history.append({"role": "user", "content": user_input})
      response = self.generate_response(user_input)
      self.conversation_history.append({"role": "assistant", "content": response})
      return response



    
   
          