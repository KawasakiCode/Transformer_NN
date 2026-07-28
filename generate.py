"""Test the model by generating text from a given prompt.

Loads transformer_weights.pth into a freshly-constructed Transformer and
autoregressively samples text from an empty context.

How to safely change parameters:
- vocab_size, n_embd, n_head, head_size, num_blocks, mlp_hidden, and
  block_size here MUST exactly match the config used in main.py for the run
  that produced transformer_weights.pth. There is no config saved alongside
  the weights file, so this has to be kept in sync BY HAND - a mismatch shows
  up as a "size mismatch" RuntimeError from load_state_dict. Whenever you
  change the model config in main.py and start a new training run, copy the
  same values here before generating from the new checkpoint.
- NOTE: as of the current main.py, training uses the GPT-2 BPE tokenizer
  (tiktoken, vocab_size=50257) directly - the vocab.json/char-level decode
  path below is left over from the earlier character-level model and will
  NOT match a checkpoint trained by the current main.py. Swap the vocab
  loading here for tiktoken.get_encoding("gpt2").decode(...) before running
  generate.py against a checkpoint trained with the current main.py config.
- The `_orig_mod.` prefix stripped below is only present because main.py
  saves state_dict() from a torch.compile()-wrapped model; if that ever
  changes (e.g. saving model._orig_mod.state_dict() instead), this stripping
  becomes a no-op and can be left in place safely either way.
"""

import json
import torch
from network import Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('vocab.json', 'r') as f:
    vocab = json.load(f)

vocab_size = vocab['vocab_size']
itos = {int(i): ch for i, ch in vocab['itos'].items()}
decode = lambda l: ''.join(itos[i] for i in l)

block_size = 256
n_embd = 304
n_head = 8
head_size = 32
num_blocks = 6
mlp_hidden = 896

model = Transformer(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size,
                     num_blocks=num_blocks, n_head=n_head, head_size=head_size, mlp_hidden=mlp_hidden)
state_dict = torch.load('transformer_weights.pth', map_location=device)
state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated_integers = model.generate(context, max_new_tokens=500, block_size=block_size)

generated_list = generated_integers[0].tolist()

print(decode(generated_list))
