import torch
import torch.nn as nn # neural network module from pytorch
from transformers import AutoModel # Use a pretrained model called BERT in order to read text

# By inheriting from the nn module, we tell our machine that this is a neural network and to treat it as such
class MultiLabelClassificationModel (nn.Module):
  # The __init__ method runs only once when we turn on the machine for the first time, it sets up all components. 
  def __init__(self, encoder_name, num_labels, dropout=0.2): #encorder_name = name of the pretrained model, num_labels = disease list, dropout = prevent cheating/overfitting
    super().__init__()
    self.encoder = AutoModel.from_pretrained(encoder_name) # loads the pre-trained transformer model
    hidden_size = self.encoder.config.hidden_size # Represents the complexity the model can handle
    self.pool = lambda x: x.last_hidden_state[:, 0, :] # The transformer reads every token in the symptom text. This line defines a simple rule to take the output for only the first token. That single vector is used as the summary of the entire symptom list. 
    self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels)) # This is essentially the disease predictor, a custom set of layers build on top of the encoder, one to prevent overfitting, and another to map the hidden features to the number of diseases

  def forward(self, input_ids, attention_mask): # This defines the process, taking the two required inputs that your dataset file prepared: input_ids (numerical symptom tokens) and attention_mask (which tells the model whats real text and whats padding)
    output = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask
    ) # Outputs a deep understanding of the text
    pooled_output = self.pool(output) # Pull out the single feature vector that best represents the entire set of symptoms
    logits = self.head(pooled_output) # Converts info into predictions
    return logits # return the predictions