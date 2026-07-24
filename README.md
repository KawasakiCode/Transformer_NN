# transformer

A character-level GPT-style transformer built from scratch with PyTorch, including a custom multi-head self-attention implementation. Trained on the Tiny Shakespeare dataset to generate Shakespeare-like text.

## Architecture

- Decoder-only transformer (`network.py`)
- Token + positional embeddings
- 6 transformer blocks, each with:
  - Custom multi-head causal self-attention (`attention.py`) — masked scaled dot-product attention implemented from scratch (no `nn.MultiheadAttention`)
  - MLP feed-forward layer (4x expansion, ReLU)
  - Pre-norm residual connections (`LayerNorm`)
- Final `LayerNorm` + linear head to vocab logits

### Hyperparameters

| Param        | Value |
|--------------|-------|
| `n_embd`     | 384   |
| `n_head`     | 6     |
| `num_blocks` | 6     |
| `block_size` | 256   |
| `batch_size` | 64    |
| `vocab_size` | 65 (character-level) |
| optimizer    | AdamW, lr=1e-3 |
| `max_iters`  | 5000  |

## Files

- `data.py` — loads `data.txt`, builds the character vocab, and creates train/val batches
- `attention.py` — `Head` and `MultiHeadAttention` modules
- `network.py` — `Transformer` model definition, forward pass, and autoregressive `generate`
- `main.py` — training loop, periodic loss estimation, saves weights to `transformer_weights.pth`
- `generate.py` — loads saved weights and generates sample text from an empty context

## Usage

Train the model:

```bash
python main.py
```

Generate text from the trained weights:

```bash
python generate.py
```

## Data

### Step 1: Tiny Shakespeare

`data.txt` is the [Tiny Shakespeare](https://github.com/karpathy/char-rnn) dataset, split 90% train / 10% validation.

### Step 2: Tiny Stories (subset)

To reduce overfitting, the model was retrained on a subset of the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset instead. `generate_tinystories_dataset()` in `data.py` downloads the dataset via Hugging Face `datasets`, extracts the first 100,000 stories to `tinystories.txt`, builds a character-level vocab (saved to `vocab.json`), and splits it 90% train / 10% validation — same character-level tokenization approach as step 1.

## Results

### Step 1: Tiny Shakespeare (5000 iterations)

- Final training loss: **0.1866**
- Lowest validation loss: **1.5656**
- Final validation loss: **3.79**

The growing gap between training and validation loss indicates the model overfit the training data over the course of training.

### Step 2: Tiny Stories subset (5000 iterations)

- Final training loss: **0.7097**
- Final validation loss: **0.7063**

Validation loss kept dropping through the end of training, with no sign of overfitting — the larger, more diverse dataset generalizes much better than Tiny Shakespeare.

## Architecture Ablation (~15M param budget)

Testing which architectural dimension (embedding size, attention heads, depth, MLP layers) contributes most to model quality, holding total parameters roughly fixed at ~15M. `attention.py`/`network.py` were refactored so `head_size` is independent of `n_embd`/`n_head` (previously tied via `n_embd // n_head`), letting attention width be detached from the residual stream width.

### Test A: wide & shallow (5000 iterations, TinyStories subset)

Config: `n_embd=784, n_head=6, head_size=130, num_blocks=2` — **15,117,198 parameters**

- Final training loss: **0.7454**
- Final validation loss: **0.7533**
