

import torch
import torch.nn as nn


class SymptomNetV2(nn.Module):
    """
    Simple fully-connected neural network for symptom  disease prediction.

    - num_features: number of input symptoms 
    - num_classes: number of diseases 
    - hidden_sizes: list defining hidden layer sizes, [512, 256, 128]
    - dropout: dropout probability after each hidden layer
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_sizes=None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        in_dim = num_features

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        # Final layer  logits for each disease
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_features)
        returns: (batch_size, num_classes) logits
        """
        return self.net(x)
