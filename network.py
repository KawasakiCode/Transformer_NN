import torch.nn as nn
import torch 
from torch.nn import functional as F

from attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_blocks):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(  
            *[Block(n_embd, n_head=6, block_size=block_size) for _ in range(num_blocks)]
        )

        # Final norm
        self.ln_f = nn.LayerNorm(n_embd)

        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        token_positional_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = token_embeddings + token_positional_embeddings

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.ln_head(x)

        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            
            # logits becomes (B, C)
            logits = logits[:, -1, :] 
            
            probs = F.softmax(logits, dim=-1) 
            
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        
        # The two core components
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = MLP(n_embd)
        
        # The Stabilizers (Layer Normalization)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 1. Communication Phase (with Residual Connection)
        x = x + self.sa(self.ln1(x))
        
        # 2. Computation Phase (with Residual Connection)
        x = x + self.ffwd(self.ln2(x))
        
        return x
