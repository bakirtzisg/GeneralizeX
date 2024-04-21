import torch.nn as nn
from typing import Sequence

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layer_dims: Sequence[int] = [7,7]):
        super().__init__()
        assert len(hidden_layer_dims) == 2, "There is 1 hidden layer, so len(hidden_layer_dims) == 2!"
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_layer_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_dims[0], hidden_layer_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_dims[1], out_features)
        )

    def forward(self, x):
        return self.layers(x)