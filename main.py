from network import Transformer
from data import generate_tinystories_dataset, get_batch
import torch
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

@torch.no_grad()
def estimate_loss(train_data, test_data, model, block_size, batch_size):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        eval_iters = 20
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(train_data, test_data, split, block_size, batch_size)
            X = X.to('cuda' if torch.cuda.is_available() else 'cpu')
            Y = Y.to('cuda' if torch.cuda.is_available() else 'cpu')
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean().item()
        
    model.train()
    return out

if __name__ == "__main__":
    train_data, test_data, vocab_size = generate_tinystories_dataset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tiny shakepear dataset
    # train_data, test_data, vocab_size = generate_data()

    train_data = train_data.to(device)
    test_data = test_data.to(device)

    block_size = 256
    batch_size = 64

    n_embd = 512
    n_head = 16
    head_size = 32
    num_blocks = 12

    model = Transformer(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, num_blocks=num_blocks, n_head=n_head, head_size=head_size)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model compiles")
    model = torch.compile(model)
    print("Compilation complete")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scaler = torch.amp.GradScaler('cuda')

    max_iters = 15001
    prev_val_loss = 10

    for iter in tqdm(range(max_iters)):
        x, y = get_batch('train', train_data, test_data, block_size, batch_size)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
          logits, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if iter % 1000 == 0:
            losses = estimate_loss(train_data, test_data, model, block_size, batch_size)
            if losses['val'] > prev_val_loss:
                print(f"Early stop step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                break
            prev_val_loss = losses['val']

            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    torch.save(model.state_dict(), 'transformer_weights.pth')
    print("Training complete")

