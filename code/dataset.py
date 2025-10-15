# This file will handle the csv dataset such that it can be fed into the model
import pandas as pd
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MultiLabelTextDataset(Dataset):
  def __init__(self, csv_path, tokenizer_name, label_list, max_length=256):
    # load the csv file
    df = pd.read_csv(csv_path)
    
    # get all the symptoms columns in a list
    symptom_columns = [col for col in df.columns if col.startswith("Symptom")]

    # fill all empty symptom cells with empty strings and join them with commas
    df["Symptoms_Combined"] = df[symptom_columns].fillna("").astype(str).agg(lambda x: ', '.join([val for val in x if val]), axis = 1) # axis = 1 means for each column

    # use this newly created column as input for the model
    self.texts = df["Symptoms_Combined"].tolist()

    # get the label list
    self.label_list = label_list
    # create a dictonary to easily map a disease to a number
    self.label_to_index = {}
    for i, label in enumerate(label_list): 
      self.label_to_index[label] = i

    self.labels = [self.encode_labels(s) for s in df["Disease"].fillna("").astype(str).to_list]

    # initialize the tokenizer and the max length
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = True)
    self.max_length = max_length

    def encode_labels(self, label_str):
      # one hot vector encoding based on the label list
      # remove any whitespaces
      label = label_str.strip()
      #initializing a zero vector 
      vec = [0]*len(self.label_list)
      vec[self.label_to_index[label]] = 1

      return vec

    def __len__(self):
      return len(self.texts)
    
    def __getitem__(self, index):
      text = self.texts[index]

      encoded_input = self.tokenizer(
          text, 
          truncation=True, 
          padding="max_length",
          max_length=self.max_length, 
          return_tensors="pt" # "pt" -> PyTorch tensors
      )
      
      item = {}
      for key, tensor in encoded_input.items():
          item[key] = tensor.squeeze(0)
      
      item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
      
      return item
    