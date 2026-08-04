"""Training script.

Loads the TinyStories dataset (data.py), builds a Transformer (network.py),
and trains it with AMP + gradient scaling + gradient accumulation, saving the
final weights to transformer_weights.pth. Also does periodic train/val loss
estimation and stops early if val loss rises between checkpoints.

How to safely change parameters:
- n_embd, n_head, head_size, num_blocks, mlp_hidden: passed straight through
  to Transformer (see network.py for how these interact and drive parameter
  count/compute). head_size is independent of n_embd/n_head - no divisibility
  constraint.
- block_size must match across training and any later generate.py run against
  the resulting checkpoint.
- micro_batch * gradient_accumulation_steps is the EFFECTIVE batch size the
  optimizer actually sees per step. Lower micro_batch (keeping the product
  fixed) if you hit CUDA out-of-memory; the loss is divided by
  gradient_accumulation_steps before backward() specifically so the
  accumulated gradient magnitude matches a single batch of that effective size.
- max_iters is a ceiling, not a target - the early-stopping check (every 1000
  iters) breaks the loop as soon as val loss increases versus the previous
  checkpoint, so raising max_iters mainly matters for architectures that are
  still improving late in training (e.g. deeper configs).
- Increasing n_embd/num_blocks/n_head raises both VRAM usage and wall-clock
  time per iteration - deeper configs (more num_blocks) in particular were
  observed to take ~10x longer than shallow ones at a similar parameter count,
  since more sequential layers must run per forward/backward pass.
- torch.compile() will re-trace/recompile whenever the model architecture or
  input shapes change, adding startup latency (visible as a long pause before
  the first iterations tick over) - this is a one-time cost, not a sign
  training is stuck.
"""

from network import Transformer
from data import generate_fineweb_edu_dataset, generate_tinystories_dataset, get_batch
import torch
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

@torch.no_grad()
def estimate_loss(train_data, test_data, model, block_size, batch_size, micro_batch):
    out = {}
    model.eval()

    # batch_size stays the logical eval batch size; micro_batch caps how many
    # sequences are actually materialized on the GPU at once (avoids OOM),
    # accumulated back up to batch_size worth of samples per eval_iter.
    accumulation_steps = batch_size // micro_batch

    for split in ['train', 'val']:
        eval_iters = 20
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            micro_losses = torch.zeros(accumulation_steps)

            for m in range(accumulation_steps):
                X, Y = get_batch(train_data, test_data, split, block_size, micro_batch)
                X = X.to('cuda' if torch.cuda.is_available() else 'cpu')
                Y = Y.to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, loss = model(X, Y)
                micro_losses[m] = loss.item()

            losses[k] = micro_losses.mean()

        out[split] = losses.mean().item()

    model.train()
    return out

if __name__ == "__main__":
    train_data, test_data, vocab_size = generate_fineweb_edu_dataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    block_size = 1024
    batch_size = 64

    micro_batch = 4
    gradient_accumulation_steps = 16

    n_embd = 1024
    n_head = 16
    head_size = 32
    num_blocks = 24

    model = Transformer(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, num_blocks=num_blocks, n_head=n_head, head_size=head_size)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    #model = torch.compile(model)


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scaler = torch.amp.GradScaler('cuda')

    max_iters = 1250000
    prev_val_loss = 20 # needs to be higher than starting val loss
    
    # after how many attempts early stop triggers
    patience = 0

    for iter in tqdm(range(max_iters)):
      x, y = get_batch(train_data, test_data, 'train', block_size, micro_batch)
      optimizer.zero_grad(set_to_none=True)

      with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
        if (iter + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        if iter % 1000 == 0:
            losses = estimate_loss(train_data, test_data, model, block_size, batch_size, micro_batch)
            if losses['val'] < prev_val_loss:
                #save best model
                torch.save(model.state_dict(), 'transformer_weights.pth')
                patience = 0
            if losses['val'] > prev_val_loss:
                if patience >= 4:
                  print(f"Early stop step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                  break
                else: 
                    patience += 1
            prev_val_loss = losses['val']

            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print("Training complete")

