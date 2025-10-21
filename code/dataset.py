# This file will handle the csv dataset such that it can be fed into the model
import torch
from torch.utils.data import Dataset
import numpy as np


class MultiLabelTextDataset(Dataset):
  NUM_FEATURES = 10 # matches the INPUT_FEATURES in model.py

  # dataset_csv_path: represents the path to where we can access our dataset
  # tokenizer_name: placeholder for a tokenizer name
  # label_list: list of the all the 41 unique diseases/labels
  # max_length: normally the max tokenized sequence length
  def __init__(self, dataset_csv_path, tokenizer_name, label_list, max_length):
    super().__init__()
    # Use a fixed data size for the synthetic data
    self.data_size = 10000 # the number of samples (one patient/feature vector)
    self.num_labels = len(label_list) # the number of lables/diseases
  
  def __len__(self):
    return self.data_size #c number of samples in the dataset

  def __getitem__(self, index): # we generate the features vector (size 10) and the one hot labels vector (size 41) corresponding to the correct disease
    # ----- GENERATING SAMPLE -----
    # Determine the index of the correct disease label for this sample which will be evaluated against the guessed label
    true_label_index = index % self.num_labels # makes each sample correspond to one label index
    
    # creates a tensor of 10 float features (input vector), initialized to 0
    features = torch.zeros(self.NUM_FEATURES, dtype=torch.float)
    # defining the maximum index of the label_list
    max_index = self.num_labels - 1 
    # assigning each feature/row to a specific disease number (the first value fo the feature vector)
    features[0] = float(true_label_index) / max_index
    
    # Fill the rest of the features with random noise
    features[1:] = 0.01 * torch.randn(self.NUM_FEATURES - 1)
    
    # Create the labels (one-hot vector for multi-label classification)
    labels = torch.zeros(self.num_labels, dtype=torch.float)
    labels[true_label_index] = 1.0 # sets the correct disease label to 1.0

    return {
      'features': features, 
      'labels': labels
    }
