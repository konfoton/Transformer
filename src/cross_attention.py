import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from LoRa import LoRa   

class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, key_query_second_dim, value_second_dim, lora = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_dim = key_query_second_dim
        self.v_dim = value_second_dim
        self.lora = lora

        key_lin   = nn.Linear(embed_dim, num_heads * self.k_dim, bias=False)
        query_lin = nn.Linear(embed_dim, num_heads * self.k_dim, bias=False)
        value_lin = nn.Linear(embed_dim, num_heads * self.v_dim, bias=False)
        out_lin   = nn.Linear(num_heads * self.v_dim, embed_dim, bias=False)

        if self.lora:
            key_lin   = self.lora.wrap(key_lin,   "key")
            query_lin = self.lora.wrap(query_lin, "query")
            value_lin = self.lora.wrap(value_lin, "value")
            out_lin   = self.lora.wrap(out_lin,   "output")

        self.key = key_lin
        self.query = query_lin
        self.value = value_lin
        self.output = out_lin
        
    def forward(self, x):
        B, L, E = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        
        k = k.view(B, L, self.num_heads, E // self.num_heads).transpose(1, 2) # (B, n_head, L, head_dim)
        v = v.view(B, L, self.num_heads, E // self.num_heads).transpose(1, 2) # (B, n_head, L, head_dim)
        q = q.view(B, L, self.num_heads, E // self.num_heads).transpose(1, 2) # (B, n_head, L, head_dim)

        att = F.scaled_dot_product_attention(q, k, v, is_causal=True) 

        out = att.transpose(1, 2).contiguous().view(B, L, E)
        out = self.output(out)

        return out