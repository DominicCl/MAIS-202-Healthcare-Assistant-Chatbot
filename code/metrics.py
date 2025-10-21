import torch
import numpy as np
from torch.nn.functional import sigmoid
from typing import List, Union

# needs to be revised for multi label classification, but the high accuracy scores still indicate that we are at least guessing the most correct label accurately

# takes two lists, the true disease list, and the predicted disease list and returns a tuple of the four TP, FP, TN, and FN counts
def get_tp_fp_tn_fn(true_indices: List[int], pred_indices: List[int]) -> tuple:
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # loops through all samples, comparing the true index vs the pred index
    for t, p in zip(true_indices, pred_indices):
        if t == p:
            # CORRECT PREDICTION!!!
            TP += 1 
        else:
            # INCORRECT GUESS :( (simplification: increment both F values) to be revised
            FP += 1
            FN += 1
    
    return TP, FP, TN, FN

# Accuracy
def custom_accuracy(true_indices: List[int], pred_indices: List[int]) -> float:
  TP, FP, TN, FN = get_tp_fp_tn_fn(true_indices, pred_indices)
    
  # accuracy calculation
  try:
    accuracy_val = (TP + TN) / ( (TP + FN) + (TN + FP) )
  except ZeroDivisionError:
    return 0.0 # accounts for a division by 0

  return accuracy_val

# Precision
def custom_precision(true_indices: List[int], pred_indices: List[int]) -> float:
  TP, FP, TN, FN = get_tp_fp_tn_fn(true_indices, pred_indices)
    
  # precision
  try:
    precision_val = TP / (TP + FP)
  except ZeroDivisionError:
    return 0.0
    
  return precision_val

# Recall
def custom_recall(true_indices: List[int], pred_indices: List[int]) -> float:
  """Calculates Recall (from Argmax)."""
  TP, FP, TN, FN = get_tp_fp_tn_fn(true_indices, pred_indices)
    
  # recall calculation
  try:
    recall_val = TP / (TP + FN)
  except ZeroDivisionError:
    return 0.0

  return recall_val

# F1
def custom_f1(true_indices: List[int], pred_indices: List[int]) -> float:

  precision_val = custom_precision(true_indices, pred_indices)
  recall_val = custom_recall(true_indices, pred_indices)
    
  if (precision_val + recall_val) == 0:
    f1_val = 0.0
  else:
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)

  return f1_val

# the actual class we import inside the training file
class MetricsEvaluator:
    def __init__(self, device):
        self.device = device
        
    def evaluate(self, model, data_loader, threshold=0.5):
       
        model.eval() # sets up the model for evaluation mode, for instance drops the dropout technique (not applicable for evaluation)
        all_true_indices = [] # stores all true label indices
        all_predicted_labels = [] #

        with torch.no_grad(): # dont wanna update the weights so dont calculate the gradients (gradients indicate how we shuld update our weights)
            for batch in data_loader:# iterating through the batches
                features = batch['features'].to(self.device) # feature inputs
                labels = batch['labels'].to(self.device) # True one-hot labels

                # the predictions (should be the same prediction as in the tes file, but we use it for the accuracy score, while the other is for calculating the loss (in the train file))
                logits = model(features)
                
                # We extract the single index of the best guess using argmax
                predicted_indices = torch.argmax(logits, dim=1).cpu().numpy()
                true_indices = torch.argmax(labels, dim=1).cpu().numpy()

                # Store all indices from each batch
                all_predicted_labels.extend(predicted_indices)
                all_true_indices.extend(true_indices)

        
        # Accuracy
        top_1_acc = custom_accuracy(all_true_indices, all_predicted_labels)
        
        # F1 score
        custom_f1_score = custom_f1(all_true_indices, all_predicted_labels)
        
        return top_1_acc, custom_f1_score

def run_metrics():
    print("Metrics file loaded successfully.")

if __name__ == '__main__':
    run_metrics()
