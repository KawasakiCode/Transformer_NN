"""Multi-head causal self-attention.

All heads are computed in one batched projection rather than looped over
individually. The earlier version registered one Head module per head, each
owning its own query/key/value matrices, and MultiHeadAttention ran them in a
Python loop:

    out = torch.cat([h(x) for h in self.heads], dim=-1)

That is mathematically identical to what this file now does, but it issued 3
matmuls per head per layer - 1152 across a 24-block, 16-head model - each one
a (batch*seq, n_embd) @ (n_embd, head_size) product far too small to occupy a
GPU. The card finished each in microseconds and idled waiting for the next
kernel launch, making the model launch-bound rather than compute-bound.

Folding the heads into the tensor shape instead gives 3 matmuls per layer, 72
across the model, and lets a single scaled_dot_product_attention call cover
every head at once with the head dimension riding along as batch.

Checkpoints from the per-head layout are convertible without retraining -
see convert_checkpoint.py.

How to safely change parameters:
- head_size (per-head width) is independent of n_embd/n_head. The projections
  map n_embd -> num_heads*head_size, and proj maps num_heads*head_size back to
  n_embd, so num_heads*head_size does NOT need to equal n_embd (this is what
  lets network.py sweep n_head/head_size separately from n_embd).
- block_size is accepted for signature compatibility but no longer used.
  Causal masking comes from is_causal=True rather than a registered mask
  buffer, which also removed 384 identical 1024x1024 float32 triangles
  (1.61GB) from VRAM and from every checkpoint.
- num_heads should stay small enough that num_heads*head_size doesn't blow up
  the proj layer's parameter count (it scales linearly with that product).
"""

import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        # One matrix per projection covering every head. Stacking the old
        # per-head (head_size, n_embd) matrices along dim 0 gives exactly
        # these, which is why old checkpoints convert losslessly.
        inner = num_heads * head_size
        self.query = nn.Linear(n_embd, inner, bias=False)
        self.key = nn.Linear(n_embd, inner, bias=False)
        self.value = nn.Linear(n_embd, inner, bias=False)

        # concat width = num_heads * head_size, independent of n_embd
        self.proj = nn.Linear(inner, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        # (B, T, inner) -> (B, T, heads, head_size) -> (B, heads, T, head_size)
        # Both view and transpose only reinterpret the tensor, so splitting
        # the heads out costs nothing.
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # Scales by 1/sqrt(head_size) and applies the causal mask internally.
        # Where a fused backend exists it never materialises the (B, heads, T, T)
        # attention matrix; where none does it falls back to the same maths as
        # the explicit implementation, so this is never worse.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # back to (B, T, inner) for the output projection
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)

        return self.proj(out)
