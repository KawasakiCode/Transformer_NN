"""Data loading and preprocessing
   Build the vocabulary, tokenizers and dictionaries for the dataset
   Create the X, Y training and test batches for the model
   X size: (batch_size, block_size)
   Y size: (batch_size, block_size)
"""

import pandas as pd
import cupy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read data
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all available characters from the data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# String to Integer mapping (stoi)
stoi = { ch:i for i,ch in enumerate(chars) }

# Integer to String mapping (itos)
itos = { i:ch for i,ch in enumerate(chars) }

# Encoder: takes a string, outputs a list of integers
encode = lambda s: [stoi[c] for c in s]

# Decoder: takes a list of integers, outputs a string
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the entire dataset and wrap it in a PyTorch Tensor
data = torch.tensor(encode(text), dtype=torch.long)

block_size = 64  # maximum context length for predictions
batch_size = 256 # number of parallel sequences to process

def get_batch(data):
    # Generate random starting indices in the 1D tensor
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Slice the input chunks (X)
    x = torch.stack([data[i : i + block_size] for i in ix])
    # Slice the target chunks (Y) - exactly shifted by 1 position
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    x.to(device)
    y.to(device)
    return x, y