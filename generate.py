"""Test the model by generating text from a given prompt"""
import torch 
from network import Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(vocab_size=65, n_embd=384, block_size=256, num_blocks=6)
model.load_state_dict(torch.load('transformer_weights.pth', map_location=device))
model = model.to(device)
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated_integers = model.generate(context, max_new_tokens=500, block_size = 256)

generated_list = generated_integers[0].tolist()

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

text = decode(s for s in generated_list)
print(text)