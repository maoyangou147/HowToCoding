import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class Mlp(nn.Module):
    def __init__(self, embed_dim, ratio, droprate):
        super().__init__()

        self.linear1 = nn.Linear(embed_dim, embed_dim * ratio)
        self.linear2 = nn.Linear(embed_dim * ratio, embed_dim)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
    
    def forward(self, x):
        x = self.dropout1(F.relu(self.linear1(x)))
        out = self.dropout2(self.linear2(x))

        return out


def attention(query, key, value, mask=None):
    sqrt_dim_head = np.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-1, -2)) / sqrt_dim_head
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, droprate):
        super.__init__()
        assert dim_model % num_heads == 0

        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_heads = dim_model // num_heads

        self.W_Q = nn.Linear(self.dim_model, self.dim_model)
        self.W_K = nn.Linear(self.dim_model, self.dim_model)
        self.W_V = nn.Linear(self.dim_model, self.dim_model)
        self.fc = nn.Linear(self.dim_model, self.dim_model)

        self.dropout = nn.Dropout(droprate)
    
    def forward(self, input_Q, input_K, input_V, mask=None):
        batch_size = input_Q.shape[0]

        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_heads, self.dim_heads).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.num_heads, self.dim_heads).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.num_heads, self.dim_heads).transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        
        attn = attention(Q, K, V, mask)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)
        output = self.dropout(self.fc(attn))

        return output


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, dim_model, droprate, mlp_ratio):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, dim_model, droprate)
        self.mlp = Mlp(dim_model, mlp_ratio, droprate)
        self.attn_norm = nn.LayerNorm(dim_model)
        self.mlp_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        x_attn = self.attn(x)
        x_attn = self.attn_norm(x + x_attn)

        x_mlp = self.mlp(x_attn)
        x_mlp = self.mlp_norm(x_mlp)

        return x_mlp


