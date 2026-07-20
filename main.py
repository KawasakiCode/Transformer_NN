from network import Transformer
from data import get_data
import torch

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        eval_iters = 200
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_data()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean().item()
        
    model.train()
    return out

if __name__ == "__main__":
    train_data, test_data = get_data()

    model = Transformer(vocab_size=65, n_embd=384, block_size=64, num_blocks=6)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    max_iters = 5000
    for iter in range(max_iters):
        x, y = get_data()

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 500 == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    torch.save(model.state_dict(), 'transformer_weights.pth')
    print("Training complete")

