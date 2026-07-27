"""Data loading and preprocessing
   Build the vocabulary, tokenizers and dictionaries for the dataset
   Create the X, Y training and test batches for the model
   X size: (batch_size, block_size)
   Y size: (batch_size, block_size)
"""

import torch
import json
import os
from datasets import load_dataset
import tiktoken
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(train_data, val_data, split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    # Generate random starting indices in the 1D tensor
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Slice the input chunks (X)
    x = torch.stack([data[i : i + block_size] for i in ix])
    # Slice the target chunks (Y) - exactly shifted by 1 position
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    
    return x, y

# Tiny Stories dataset
def generate_tinystories_dataset():
    file_path = "tinystories.txt"
    output_file_path = "train.bin"

    tokenizer = tiktoken.get_encoding("gpt2")

    chunk_size = 10_000_000
    all_tokens = []
    
    # 1. Skip the download if we already built the text file
    if not os.path.exists(file_path):
        print("Downloading dataset from Hugging Face...")
        dataset = load_dataset("roneneldan/TinyStories")
        
        # Open strictly for writing
        with open(file_path, "w", encoding="utf-8") as f:
            for row in dataset["train"]:
                f.write(dataset["train"][row]["text"] + "\n")
    
    print(f"Loading data from {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            text_chunk = f.read(chunk_size)
            if not text_chunk:
                break
            
            # Convert text chunk to token IDs
            ids = tokenizer.encode_ordinary(text_chunk)
            all_tokens.extend(ids)

    # Convert to uint16 NumPy array and save to disk
    np_tokens = np.array(all_tokens, dtype=np.uint16)
    np_tokens.tofile(output_file_path)

    print(f"Saved {len(np_tokens):,} tokens to {output_file_path}")
    
    data = np.memmap("train.bin", dtype=np.uint16, mode="r")
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]
    
    # 4. Return all four variables expected by main.py
    return train_data, test_data, tokenizer.n_vocab