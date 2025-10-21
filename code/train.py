import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim import Adam 
import pandas as pd
import numpy as np

# importing our other implemented classes
from dataset import MultiLabelTextDataset
from model import MultiLabelClassificationModel
from metrics import MetricsEvaluator  # Assuming this file exists and works

# ----- FLOW OF THE TRAINING PROGRAM -----
# Load dataset  →  Split into train/validation  →  Create batches  →  Train for EPOCHS (each epoch goes through all batches)


# defining our constants (hyperparameter/parameters set by the designer)
CSV_PATH = '../data/dataset.csv'
MODEL_NAME = 'MLC-Logistic-Reression-Model'
MAX_LENGTH = 128 # maximum tokenizer length
# BATCH_SIZE
BATCH_SIZE = 64
LEARNING_RATE = 0.001 
EPOCHS = 300 # number of training sessions
# Dropout remains at 0.0 for this clean, synthetic dataset.
DROPOUT_RATE = 0.01 # technique to avoid overfitting (memorization by the model), we want it to recognize general patterns
DISEASE_LIST = [f'Disease_{i}' for i in range(41)] # list of all 41 unqique diseases
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def train_model(): # this is the main function to run the training process
  # loading the dataset with out hyperparameters
  entire_dataset = MultiLabelTextDataset(
    dataset_csv_path=CSV_PATH,
    tokenizer_name=MODEL_NAME,
    label_list=DISEASE_LIST,
    max_length=MAX_LENGTH
  )

# spliting the dataset (differently each time to avoid untrue bias)
  train_size = int(0.9 * len(entire_dataset)) # 90% of the dataset 
  val_size = len(entire_dataset) - train_size # 10% of the dataset
  train_dataset, val_dataset = random_split(entire_dataset, [train_size, val_size]) # taking random parts of the dataset for both the training and validation datasets

  # now that we have the dataset loaded in a format we can work with, we are going to split the dataset into 90% training (teach the model and updates weights and biases), and 10% validation (used to test the model)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  # 16 for better efficiency?
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
  
  num_lables = len(DISEASE_LIST)

  # getting our model object 
  model = MultiLabelClassificationModel(
    encoder_name=MODEL_NAME, # BERT
    num_labels=num_lables,
    dropout=DROPOUT_RATE
  ).to(DEVICE) # sending the model to specific parts of the hardware to train it faster

  # loss function, which essentially measure the error for a classification, we aim to minimize the loss as much as the model can
  loss_func = nn.BCEWithLogitsLoss()
  evaluator = MetricsEvaluator(DEVICE) # evaluating our classifications
  # the optimizer, will update the weights according to our evaluation
  optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

  # the actual training loop
  print("Starting the training loop")
  for epoch in range(EPOCHS): 
    model.train()
    total_loss = 0 # reseting the total loss to 0

    for step, batch in enumerate(train_loader): # iterating over each batch (step is just the index of each batch)
      features = batch['features'].to(DEVICE) # pulling features and labels from the batch dictionary
      labels = batch['labels'].to(DEVICE)

      optimizer.zero_grad() # zeros out all gradients from the previous iteration

      # model prediction/classification!!!
      logits = model(features)

      # calculate the loss
      loss = loss_func(logits, labels)
      total_loss += loss.item()

      # calculating the loss gradient
      loss.backward()

      # optimizing, this will update the weights
      optimizer.step()

      # periodic logging of losses (to visualize improvement)
      # Since we should only have 1 step, we log if step is 0 (the only step)
      if step == 0:
        print(f"  The epoch is: {epoch+1}/{EPOCHS}, the step is: {step+1}/{len(train_loader)}, and the loss is: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    # evaluation after the epoch
    val_acc, val_f1 = evaluator.evaluate(model, val_loader)
    
    print(f"\n--- Epoch {epoch+1} Finished ---")
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Top-1 Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1 Score: {val_f1:.4f}\n")

    # save the model 
    torch.save(model.state_dict(), 'disease_predictor_final.pt')
    print("Training complete. Model saved as 'disease_predictor_final.pt'")


if __name__ == '__main__':
    train_model()
