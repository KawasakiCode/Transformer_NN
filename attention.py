"""Attention and Multi-Head Attention modules
   This file contains the classes and forward methods for the attention and
   multi-head attention modules used"""

import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """
    
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
            B, T, C = x.shape # Batch, Time (Sequence Length), Channels (n_embd)
            
            q = self.query(x) # (B, T, head_size)
            k = self.key(x)   # (B, T, head_size)
            v = self.value(x) # (B, T, head_size)

            wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) 
            
            # Overwrite all "future" interactions with -infinity
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            
            # -infinity turns into exactly 0.0, so future tokens get 0% attention
            wei = F.softmax(wei, dim=-1)
            
            out = wei @ v
            
            return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Run the input through every head independently
        # Then concatenate their outputs along the final dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Pass the concatenated result through the projection layer
        out = self.proj(out)
        
        return out