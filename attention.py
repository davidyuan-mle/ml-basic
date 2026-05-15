import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads 
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)        # (B, n_heads, T, d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)        # (B, n_heads, T, d_k)
        V = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)        # (B, n_heads, T, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)                    # (B, n_heads, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))                  
        weights = F.softmax(scores, dim=-1)

        out = (weights @ V).transpose(1, 2).reshape(B, T, d_model)

        return self.W_o(out), weights


# simulation 
torch.manual_seed(42)

B, T, d_model, n_heads = 2, 5, 32, 4
x = torch.randn(B, T, d_model)

mha = MultiHeadAttention(d_model, n_heads)

# 1. Self-attention (no mask)
out, weights = mha(x)
print("=== Self-Attention (no mask) ===")
print(f"Input:   {x.shape}")       # (2, 5, 32)
print(f"Output:  {out.shape}")      # (2, 5, 32)
print(f"Weights: {weights.shape}")  # (2, 4, 5, 5)
print(f"Weights sum per row: {weights[0, 0].sum(dim=-1)}")  # all 1s (valid softmax)

