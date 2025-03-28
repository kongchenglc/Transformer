import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores and scale to prevent gradient vanishing
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # Apply mask to prevent attending to certain positions
        attn_probs = F.softmax(attn_scores, dim=-1)  # Convert to probability distribution
        output = torch.matmul(attn_probs, V)  # Weight values by attention probabilities
        return output
        
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
    def forward(self, Q, K, V, mask=None):
        # Project and split inputs into multiple attention heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads back into original dimension space
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.d_model)
        
        return self.W_o(attn_output)  # Final projection to mix information from all heads
