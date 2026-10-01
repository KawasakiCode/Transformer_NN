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
import os
import math
import signal
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

def save_checkpoint(target_path, iter_num, model, optimizer, scaler, best_val_loss):
    # Write to a temp file and rename over the target, rather than saving
    # directly to it - a crash mid torch.save() would otherwise corrupt the
    # file resume depends on. Rename is atomic, so a good checkpoint is
    # never left partially overwritten.
    # fsync forces the write to actually reach the physical disk before we
    # proceed - without it, the file can look fully written (readable,
    # correct size) while still only sitting in the page cache, and an
    # abrupt container/process restart can lose it despite it having
    # appeared to save successfully.
    tmp_path = target_path + ".tmp"
    with open(tmp_path, 'wb') as f:
        torch.save({
            'iter': iter_num,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
        }, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)

def get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    """Linear warmup, then cosine decay from max_lr down to min_lr.

    Computed from the ABSOLUTE iteration rather than held as optimizer state,
    so the schedule is identical whether a run starts fresh or resumes from a
    checkpoint. A stateful torch scheduler would restart its curve at the top
    on every resume, silently undoing the decay each session.
    """
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it >= lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(train_data, test_data, model, block_size, batch_size, micro_batch):
    out = {}
    model.eval()

    # batch_size stays the logical eval batch size; micro_batch caps how many
    # sequences are actually materialized on the GPU at once (avoids OOM),
    # accumulated back up to batch_size worth of samples per eval_iter.
    accumulation_steps = batch_size // micro_batch

    for split in ['train', 'val']:
        eval_iters = 5
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

    micro_batch = 2
    gradient_accumulation_steps = 32

    n_embd = 512
    n_head = 16
    head_size = 32
    num_blocks = 24
    mlp_hidden = 3072  # 3x n_embd instead of the 4x default - trims ~355.9M -> ~305.6M params
                        # without touching depth/heads/embedding width or context length

    model = Transformer(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, num_blocks=num_blocks, n_head=n_head, head_size=head_size, mlp_hidden=mlp_hidden)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.compile(model)


    # Learning rate schedule. The constant 1e-3 this replaces was high for a
    # model this size and, with no decay, left training oscillating around a
    # minimum it could never settle into - train and val both stuck at ~3.45
    # with no gap between them. max_lr is applied via get_lr() every
    # iteration, so the lr baked into a resumed optimizer state is ignored.
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_iters = 2000

    # beta2=0.95 rather than the 0.999 default. 0.999 averages the squared
    # gradient over roughly the last 1000 steps, which is sluggish for
    # transformer pretraining where gradient scale shifts as training moves;
    # 0.95 (~20 steps) tracks it far more responsively and is the standard
    # choice for LM pretraining (GPT-3, nanoGPT). weight_decay stays at
    # AdamW's 0.01 default - see the note after the resume block.
    betas = (0.9, 0.95)
    weight_decay = 0.01

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr,
                                  betas=betas, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler('cuda')

    # max_iters is the ABSOLUTE target (one epoch over the ~10B token
    # dataset). session_iters is how many iterations this particular run
    # does before stopping cleanly - set it to roughly what fits in the time
    # the machine is left on, so the run ends on its own terms (saving a
    # final checkpoint) instead of being killed mid-write. Progress toward
    # max_iters is tracked by the checkpoint's stored iter, so there's
    # nothing to record by hand between sessions.
    max_iters = 1250000
    session_iters = 50000

    # Where the cosine curve bottoms out at min_lr - NOT necessarily
    # max_iters, but wherever you actually intend to stop. The shape of the
    # whole schedule depends on it: stop well short of lr_decay_iters and the
    # rate is still high at that point, so the model never gets the low-lr
    # settling phase where the last of the loss comes out. Set it to the
    # realistic stopping point, not the aspirational one.
    lr_decay_iters = 800000

    # Lowest val loss seen so far, across all sessions. Only ever decreases,
    # so transformer_weights.pth is overwritten solely by a genuine new best.
    best_val_loss = float('inf')

    # Checkpoint/resume: saves model + optimizer + scaler + iteration state
    # together, not just weights, so a crash doesn't lose AdamW's momentum
    # buffers or force restarting the iteration count from 0. Stored on the
    # D drive (/mnt/d) - the root partition doesn't have room for this on
    # top of the dataset.
    #
    # Two alternating files (not one) - an abrupt SIGKILL mid-write can
    # corrupt the NTFS file record of whichever file is being replaced (seen
    # in practice on this drive), and atomic rename + fsync can't fully
    # protect against the underlying filesystem driver itself being killed
    # mid metadata-transaction. Alternating means a corrupted write only
    # ever costs the last checkpoint_every iterations, since the OTHER file
    # (one cycle older) is untouched by that write and still loadable.
    checkpoint_paths = ["/mnt/d/checkpoint_a.pth", "/mnt/d/checkpoint_b.pth"]
    checkpoint_every = 500
    # How often to run estimate_loss and print train/val loss. Lower it to
    # sample the loss curve more often over a short diagnostic run; each eval
    # costs roughly a minute, so 5000 is the sensible value for long sessions.
    eval_every = 5000
    start_iter = 0

    # Ctrl+C sets this instead of interrupting immediately, so we always
    # finish the in-flight iteration's forward/backward before saving -
    # writing a checkpoint mid-backward would capture a torn, inconsistent
    # state. Checked once per iteration, right after that iteration's work
    # is done.
    stop_requested = False

    def handle_sigint(signum, frame):
        global stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    # Probe both checkpoints for their iteration number, then load only the
    # winner. Everything here uses map_location='cpu' deliberately: loading
    # with map_location=device puts each checkpoint's full ~3.2GB of tensors
    # straight into VRAM, so comparing two of them held ~6.4GB on the GPU at
    # once and OOM'd before training started. load_state_dict copies from CPU
    # to the model's device by itself, and AdamW moves loaded state onto each
    # param's device, so nothing needs to be on the GPU to begin with.
    best_path, best_iter = None, -1
    for path in checkpoint_paths:
        if not os.path.exists(path):
            continue
        try:
            probe = torch.load(path, map_location='cpu')
            if probe['iter'] > best_iter:
                best_path, best_iter = path, probe['iter']
            del probe
        except Exception as e:
            print(f"Warning: could not read checkpoint {path} ({e}), skipping it")

    if best_path is not None:
        ckpt = torch.load(best_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_iter = ckpt['iter'] + 1
        # Checkpoints written before this key was renamed store it as
        # 'prev_val_loss'; fall back so existing checkpoints still load.
        best_val_loss = ckpt.get('best_val_loss', ckpt.get('prev_val_loss', float('inf')))
        del ckpt
        torch.cuda.empty_cache()
        print(f"Resumed at iteration {start_iter} from {best_path}, best_val_loss={best_val_loss:.4f}")

    # optimizer.load_state_dict() overwrites each param_group's hyperparameters
    # with the ones stored in the checkpoint, so anything set in the AdamW
    # constructor above is silently reverted to whatever the old run used.
    # Re-apply here, after loading, or betas/weight_decay changes never take
    # effect on a resumed run. (lr escapes this because the loop sets it every
    # iteration.)
    for param_group in optimizer.param_groups:
        param_group['betas'] = betas
        param_group['weight_decay'] = weight_decay

    end_iter = min(start_iter + session_iters, max_iters)
    print(f"This session: iterations {start_iter} -> {end_iter} "
          f"({100 * start_iter / max_iters:.2f}% -> {100 * end_iter / max_iters:.2f}% of {max_iters})")

    # initial/total make the bar show absolute iteration numbers on a resume
    # (85000/1250000) instead of restarting the displayed count at 0 - the
    # loop variable itself was always absolute, only the display was relative.
    for iter in tqdm(range(start_iter, end_iter), initial=start_iter, total=end_iter):
      lr = get_lr(iter, warmup_iters, lr_decay_iters, max_lr, min_lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

      x, y = get_batch(train_data, test_data, 'train', block_size, micro_batch)

      with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
        if (iter + 1) % gradient_accumulation_steps == 0:
            # unscale_ first so clipping sees true gradient magnitudes rather
            # than the fp16-scaled ones (the scale factor is ~65536, which
            # would make every step look like it needs clipping).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if stop_requested:
            target_path = checkpoint_paths[(iter // checkpoint_every) % 2]
            save_checkpoint(target_path, iter, model, optimizer, scaler, best_val_loss)
            print(f"Interrupted - saved checkpoint at iteration {iter} to {target_path}")
            break

        if iter % checkpoint_every == 0 and iter != 0:
            # Alternates between the two paths in checkpoint_paths (see
            # comment above) so a corrupted write only ever affects the
            # older of the two, not the only copy that exists.
            target_path = checkpoint_paths[(iter // checkpoint_every) % 2]
            save_checkpoint(target_path, iter, model, optimizer, scaler, best_val_loss)

        if iter % eval_every == 0 and iter != 0:
            losses = estimate_loss(train_data, test_data, model, block_size, batch_size, micro_batch)

            # Only a new all-time best overwrites the saved weights. The old
            # code reassigned best_val_loss on every eval, so it compared
            # against the PREVIOUS eval instead of the best ever - a 3.31 model
            # could be overwritten by a later 3.33 one just because the eval
            # before it happened to be 3.35.
            improved = losses['val'] < best_val_loss
            if improved:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), 'transformer_weights.pth')

            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"lr {lr:.2e}, best {best_val_loss:.4f}{' <- saved' if improved else ''}")

    if not stop_requested:
        # Session finished its budget rather than being interrupted. The last
        # periodic save was at the previous multiple of checkpoint_every, so
        # save again here or the tail of the session is lost.
        target_path = checkpoint_paths[(iter // checkpoint_every) % 2]
        save_checkpoint(target_path, iter, model, optimizer, scaler, best_val_loss)
        print(f"Session complete - saved checkpoint at iteration {iter} to {target_path}")
        print(f"Progress: {iter}/{max_iters} ({100 * iter / max_iters:.2f}%)")