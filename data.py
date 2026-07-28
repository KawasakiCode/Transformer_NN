"""Data loading and preprocessing
   Build the vocabulary, tokenizers and dictionaries for the dataset
   Create the X, Y training and test batches for the model
   X size: (batch_size, block_size)
   Y size: (batch_size, block_size)

How to safely change parameters:
- get_batch(block_size, batch_size): block_size here MUST match the
  block_size passed to the Transformer (network.py) - it's the length of
  each training sequence. batch_size is independent of the model and can be
  tuned freely for throughput/VRAM (main.py further splits this into
  micro_batch + gradient_accumulation_steps if the full batch doesn't fit).
- generate_tinystories_dataset(): tokenizes with the GPT-2 BPE vocab via
  tiktoken (50,257 tokens) and caches results to tinystories.txt (raw text,
  downloaded once from Hugging Face) and train.bin (tokenized uint16 array).
  Delete tinystories.txt and/or train.bin to force a re-download/re-tokenize
  (e.g. after changing which/how many stories to pull) - otherwise the cached
  files are reused as-is even if you change chunk_size or the story count.
  chunk_size only affects how much raw text is read into memory at once
  while tokenizing; it doesn't change the resulting tokens.
- The returned vocab_size (tokenizer.n_vocab) must be passed into Transformer
  unchanged - it's fixed by the tokenizer, not a tunable hyperparameter.
"""

import torch
import os
from datasets import load_dataset
import tiktoken
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(train_data, val_data, split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    # Generate random starting indices in the 1D tensor
    ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device)

    offsets = torch.arange(block_size, device=data.device)
    indices = ix.unsqueeze(1) + offsets

    # Extract sequences on CUDA and cast to long for nn.Embedding
    x = data[indices].long()
    y = data[indices + 1].long()
    
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
                f.write(row["text"] + "\n")
    
    print(f"Loading data from {file_path}...")

    if not os.path.exists("train.bin"):
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
    
    raw_data = np.fromfile("train.bin", dtype=np.uint16)
    
    n = int(0.9 * len(raw_data))
    train_data = (
        torch.from_numpy(raw_data[:n].astype(np.int32)).to(device).contiguous()
    )
    test_data = (
        torch.from_numpy(raw_data[n:].astype(np.int32)).to(device).contiguous()
    )
    
    return train_data, test_data, tokenizer.n_vocab