"""Decoder-only transformer model definition.

Defines Transformer (token+positional embeddings -> stack of Blocks -> final
norm -> vocab logits), Block (attention + MLP with pre-norm residuals), and
MLP (the feed-forward sublayer).

How to safely change parameters (all passed into Transformer(...) from main.py):
- n_embd, num_blocks: the two dimensions that actually drive parameter count
  (roughly params ~ 12 * n_embd^2 * num_blocks). Raising either increases both
  compute and VRAM use significantly since the scaling is quadratic in n_embd.
- n_head, head_size: head_size is INDEPENDENT of n_embd/n_head here (the proj
  layer in attention.py maps num_heads*head_size -> n_embd), so you can freely
  pick any n_head/head_size combination without needing n_embd % n_head == 0.
  If head_size is left as None, it falls back to the old tied behavior
  (n_embd // n_head).
- mlp_hidden: width of the MLP's hidden layer, defaults to 4 * n_embd. Safe to
  change independently of n_embd/num_blocks/n_head.
- block_size must match the block_size used when building the training batches
  (data.py) and stay fixed for a given set of saved weights - it sizes the
  causal mask buffer (attention.py) and the positional embedding table.
- vocab_size must match whatever tokenizer/vocab produced your training data
  (see data.py). Mismatches here or in the above show up as size-mismatch
  errors when loading a saved state_dict (see generate.py).
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_blocks,
                 n_head=6, head_size=None, mlp_hidden=None):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head=n_head, block_size=block_size,
                    head_size=head_size, mlp_hidden=mlp_hidden)
             for _ in range(num_blocks)]
        )

        # Final norm
        self.ln_f = nn.LayerNorm(n_embd)

        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        token_positional_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = token_embeddings + token_positional_embeddings

        # Gradient checkpointing: only during training, since it trades
        # recompute for memory (each block's activations are discarded after
        # its forward and recomputed during backward instead of kept
        # resident) - there's no backward pass at eval time, so no point
        # paying the recompute cost then.
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)

        if targets is None:
            logits = self.ln_head(x)
            return logits, None

        # Chunked loss: never materialize the full (B*T, vocab_size) logits
        # tensor at once - with vocab_size in the tens of thousands, that
        # tensor is a large, fixed memory cost independent of model size or
        # gradient checkpointing (which only covers the block stack above,
        # not this final projection). Chunking the sequence dimension keeps
        # only one small slice of logits alive at a time.
        x_flat = x.view(B * T, -1)
        targets_flat = targets.view(B * T)

        chunk_size = 1024
        total_loss = 0.0
        total_count = 0

        for start in range(0, x_flat.size(0), chunk_size):
            chunk_targets = targets_flat[start:start + chunk_size]
            chunk_logits = self.ln_head(x_flat[start:start + chunk_size])
            total_loss = total_loss + F.cross_entropy(chunk_logits, chunk_targets, reduction='sum')
            total_count += chunk_targets.size(0)

        loss = total_loss / total_count

        # No caller currently uses the logits when targets is provided
        # (only loss) - skip materializing/returning the full tensor.
        return None, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -block_size:]
            
            logits, _ = self(idx_cond)
            
            # logits becomes (B, C)
            logits = logits[:, -1, :] 
            
            probs = F.softmax(logits, dim=-1) 
            
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx


class MLP(nn.Module):
    def __init__(self, n_embd, hidden=None):
        super().__init__()
        hidden = hidden or 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, head_size=None, mlp_hidden=None):
        super().__init__()
        # head_size is independent of n_embd/n_head - defaults to the tied
        # n_embd // n_head split only when not given explicitly
        head_size = head_size or (n_embd // n_head)

        # The two core components
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = MLP(n_embd, hidden=mlp_hidden)
        
        # The Stabilizers (Layer Normalization)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 1. Communication Phase (with Residual Connection)
        x = x + self.sa(self.ln1(x))
        
        # 2. Computation Phase (with Residual Connection)
        x = x + self.ffwd(self.ln2(x))
        
        return x
