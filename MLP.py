import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, embed_dim, d_mlp):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, d_mlp)
        self.fc2 = nn.Linear(d_mlp, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x