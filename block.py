import torch
import torch.nn as nn
import cross_attention
import MLP
class Block(nn.Module):
    
    def __init__(self, embed_dim, num_heads, d_mlp, key_query_second_dim, value_second_dim):
        super().__init__()
        self.cross_attention = cross_attention.CrossAttention(embed_dim, num_heads, key_query_second_dim, value_second_dim)
        self.mlp = MLP.MLP(embed_dim, d_mlp)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.cross_attention(x))
        x = self.norm2(x + self.mlp(x))
        return x