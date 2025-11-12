import torch.nn as nn

class MultiLabelClassificationModel(nn.Module):
    def __init__(self, encoder_name, num_labels, input_features, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, features):
        return self.network(features)


