import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PositionalEncodingFernando(nn.Module):
    pass


class MultiHeadAttentionFernando(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.d_k = self.embed_dim // self.n_heads

        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)

        self.linear_out = nn.Linear()

    def forward(self, x):
        q, k, v = x, x, x


















class PositionalEncodingBencmark(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingBencmark, self).__init__()
        
        # Create a long tensor of max_len positions
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Compute the positional encodings using sin and cos
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer so it’s saved in the model state_dict but not optimized
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    

class MultiHeadAttentionBenchmark(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionBenchmark, self).__init__()
        assert d_model % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V transformations
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Perform linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc_out(attn_output)