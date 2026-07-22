from network import Transformer
from data import generate_data, generate_tinystories_dataset, get_batch
import torch
from tqdm import tqdm

@torch.no_grad()
def estimate_loss(train_data, test_data, model, block_size, batch_size):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        eval_iters = 200
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
    # train_data, test_data, vocab_size = generate_data()

    block_size = 256
    batch_size = 64

    model = Transformer(vocab_size=vocab_size, n_embd=384, block_size=block_size, num_blocks=6)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    max_iters = 5000
    for iter in tqdm(range(max_iters)):
        x, y = get_batch('train', train_data, test_data, block_size, batch_size)
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 500 == 0:
            losses = estimate_loss(train_data, test_data, model, block_size, batch_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    torch.save(model.state_dict(), 'transformer_weights.pth')
    print("Training complete")

