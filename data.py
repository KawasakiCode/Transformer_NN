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
- generate_fineweb_edu_dataset(): streams HuggingFaceFW/fineweb-edu
  (sample-10BT, ~10B GPT-2 tokens) and tokenizes incrementally, flushing to
  fineweb_train.bin in bounded chunks - never holds the full token stream in
  memory (unlike generate_tinystories_dataset's all_tokens list, which only
  works because TinyStories is small). The cached .bin is then memory-mapped
  rather than loaded as a dense tensor, since 10B tokens (~40GB as int32)
  doesn't fit in VRAM or comfortably in RAM - get_batch() only ever
  materializes the small sampled micro-batch as a real GPU tensor. Requires
  an HF_TOKEN env var with at least read access (see huggingface.co/settings/tokens).
"""

import torch
import os
import queue
import threading
from datasets import load_dataset
import tiktoken
import numpy as np
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(train_data, val_data, split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    # Generate random starting indices in the 1D tensor
    ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device)
    offsets = torch.arange(block_size, device=data.device)
    indices = ix.unsqueeze(1) + offsets
    # Extract sequences and cast to long for nn.Embedding. Explicitly moved
    # to `device` here (not just relying on data already living there) since
    # data may be CPU-resident/memory-mapped for large datasets - only this
    # small extracted batch, not the whole dataset, needs to be a GPU tensor.
    x = data[indices].long().to(device)
    y = data[indices + 1].long().to(device)

    return x, y


class BatchPrefetcher:
    """Fetch batches on a worker thread so disk reads overlap with GPU compute.

    get_batch() reads random windows out of a memory-mapped multi-GB dataset,
    which on this setup measured 400-800ms per call - roughly a third to a
    half of each training iteration - with high variance from disk latency.
    Because the training loop called it inline, none of that overlapped with
    compute: the GPU finished a batch, then sat idle waiting for the next one,
    which showed up as utilisation oscillating between 100% and 30%.

    A worker thread keeps a small queue of ready batches topped up while the
    main thread trains, so an iteration costs max(compute, fetch) instead of
    compute + fetch. The batches are identical to what get_batch would have
    returned - same sampling, same shapes - they are just produced earlier.

    Only worth using for the training loop. estimate_loss runs rarely and can
    keep calling get_batch directly.
    """

    def __init__(self, train_data, val_data, split, block_size, batch_size, depth=4):
        self._args = (train_data, val_data, split, block_size, batch_size)
        self._queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self):
        while not self._stop.is_set():
            batch = get_batch(*self._args)
            # Timeout rather than a blocking put, so a stopped prefetcher
            # cannot wedge on a full queue nobody is draining.
            while not self._stop.is_set():
                try:
                    self._queue.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next_batch(self):
        return self._queue.get()

    def close(self):
        self._stop.set()
        # Drain so the worker's pending put() can complete and the thread exit.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass


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

# FineWeb-Edu dataset (sample-10BT, ~10B GPT-2 tokens)
def _assert_dataset_dir_usable(output_file_path, needed_gb=50):
    """Refuse to start a multi-hour tokenization into the wrong filesystem.

    A missing dataset file is indistinguishable from a drive that failed to
    mount: either way os.path.exists() returns False and tokenizing begins.
    This has gone wrong four times. The specific failure is that udisks2
    mounts the drive under a suffixed name (loukoumades1) when a stale
    mountpoint directory already exists, leaving the Docker bind mount
    pointing at that stale directory - which lives on /run, a 3GB tmpfs.
    Tokenization then writes tens of GB into RAM until it fills, taking
    the machine's runtime directory down with it.

    A free-space check catches exactly that case, since no plausible
    destination for this dataset has less than 50GB free.
    """
    directory = os.path.dirname(output_file_path) or '.'

    if not os.path.isdir(directory):
        raise RuntimeError(
            f"{directory} does not exist. The dataset drive is probably not "
            f"mounted - refusing to tokenize into a path that isn't there.")

    stat = os.statvfs(directory)
    free_gb = stat.f_bavail * stat.f_frsize / 1e9

    if free_gb < needed_gb:
        raise RuntimeError(
            f"{directory} has only {free_gb:.1f}GB free, need ~{needed_gb}GB.\n"
            f"This almost always means the drive is not mounted and {directory} "
            f"is a stale directory on a small filesystem (often tmpfs).\n"
            f"Check with:  mount | grep loukoumades\n"
            f"If it shows a suffixed name like 'loukoumades1', unmount it, "
            f"remove the stale directory, and remount so the bind mount "
            f"resolves to the real drive.")

    print(f"Dataset directory {directory}: {free_gb:.0f}GB free - OK")


def generate_fineweb_edu_dataset():
    output_file_path = "/mnt/d/fineweb_train.bin"

    tokenizer = tiktoken.get_encoding("gpt2")

    if not os.path.exists(output_file_path):
        # Only reached when the dataset appears absent, which is exactly when
        # a mis-resolved mount would silently send a 40GB write into RAM.
        _assert_dataset_dir_usable(output_file_path)
        print("Streaming FineWeb-Edu (sample-10BT) from Hugging Face...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )

        # Flush tokens to disk in bounded chunks instead of accumulating one
        # giant Python list (as generate_tinystories_dataset does) - at 10B
        # tokens that list would exhaust RAM long before finishing.
        flush_every = 1_000_000
        token_buffer = []
        total_tokens = 0

        with open(output_file_path, "wb") as f:
            for row in tqdm(dataset):
                token_buffer.extend(tokenizer.encode_ordinary(row["text"]))

                if len(token_buffer) >= flush_every:
                    np.array(token_buffer, dtype=np.uint16).tofile(f)
                    total_tokens += len(token_buffer)
                    token_buffer = []

            if token_buffer:
                np.array(token_buffer, dtype=np.uint16).tofile(f)
                total_tokens += len(token_buffer)

        print(f"Saved {total_tokens:,} tokens to {output_file_path}")

    # Memory-map instead of loading as a dense array - at 10B tokens this is
    # ~20GB on disk (~40GB if densified to int32), far bigger than available
    # VRAM. get_batch() only pulls small sampled windows from this, so pages
    # are read from disk on demand rather than all at once.
    raw_data = np.memmap(output_file_path, dtype=np.uint16, mode='r')

    n = int(0.9 * len(raw_data))
    train_data = torch.from_numpy(raw_data[:n])
    test_data = torch.from_numpy(raw_data[n:])

    return train_data, test_data, tokenizer.n_vocab
