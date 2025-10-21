import torch
import torch.nn as nn 
# NOTE: Transformer imports are removed as this model uses simple numerical features.

# The number of input features must match the constant in dataset.py (which is 10)
INPUT_FEATURES = 10 

class MultiLabelClassificationModel (nn.Module): # inheriting from the nn module
  
  # Sets up the layers for the neural network
  def __init__(self, encoder_name, num_labels, dropout=0.2): 
    super().__init__()
    
    # We ignore 'encoder_name' as we are not using a Transformer, but keep 
    # the argument signature with train.py.

    # sequential makes it so we flow through each layer one at a time
    # Flow: 10 features 
    #  ↓
    # Expand to 128 ← Learn complex patterns for better guesses
    #  ↓
    # Shrink back to 41 ← Get disease predictions
    self.network = nn.Sequential(
        # takes the input features and applies the Wx + b equation over 128 columns (weight matrix = 10X128)
        nn.Linear(INPUT_FEATURES, 128),  
        nn.ReLU(),                       
        nn.Dropout(dropout),  # dropout some neurons           
        # projects the hidden layer to 41 output classes (logits)
        nn.Linear(128, num_labels)       
    )

  def forward(self, features): 
   
    logits = self.network(features)
    
    return logits # return the raw prediction scores (logits)
