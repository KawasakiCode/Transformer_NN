"""Test the model by generating text from a given prompt"""
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
