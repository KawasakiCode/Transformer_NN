"""Data loading and preprocessing
   Build the vocabulary, tokenizers and dictionaries for the dataset
   Create the X, Y training and test batches for the model
   X size: (batch_size, block_size)
   Y size: (batch_size, block_size)
"""

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data():
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 2. Convert the ENTIRE text into a single CPU tensor
    # This will take up a few megabytes of standard RAM, which is nothing.
    data = torch.tensor(encode(text), dtype=torch.long)

    # 3. Create the Train/Validation splits (90% / 10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size

def get_batch(train_data, val_data, split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    # Generate random starting indices in the 1D tensor
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Slice the input chunks (X)
    x = torch.stack([data[i : i + block_size] for i in ix])
    # Slice the target chunks (Y) - exactly shifted by 1 position
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y.to('cuda' if torch.cuda.is_available() else 'cpu')
    return x, y