"""Sample text from a trained checkpoint.

Loads transformer_weights.pth (the best-val weights saved by main.py) into a
freshly-constructed Transformer and autoregressively continues a prompt.

What this model does and doesn't do:
- It is a raw next-token predictor trained on FineWeb-Edu web text. It has had
  no instruction tuning, so it CONTINUES text rather than answering it. A
  prompt phrased as a question tends to produce more questions, or page
  furniture like navigation links, because that is what follows a question in
  its training data.
- Prompts that work are openings it can plausibly continue: the first half of
  a sentence, or the opening line of an explanatory article.

How to safely change parameters:
- block_size, n_embd, n_head, head_size, num_blocks and mlp_hidden MUST match
  the values in main.py for the run that produced the weights. Nothing is
  saved alongside the weights to enforce this, so it is kept in sync BY HAND -
  a mismatch surfaces as a "size mismatch" RuntimeError from load_state_dict.
- The `_orig_mod.` prefix stripped below exists because main.py saves
  state_dict() from a torch.compile()-wrapped model. Harmless either way.

Usage:
    python3 generate.py "The process of photosynthesis"
    python3 generate.py "In 1492," --tokens 200 --temperature 0.7 --samples 3
"""

import argparse
import torch
import tiktoken
from network import Transformer

# Must match main.py for the run that produced transformer_weights.pth
block_size = 1024
n_embd = 512
n_head = 16
head_size = 32
num_blocks = 24
mlp_hidden = 3072

parser = argparse.ArgumentParser()
parser.add_argument('prompt', nargs='?', default="The process of photosynthesis",
                    help="text for the model to continue")
parser.add_argument('--tokens', type=int, default=300,
                    help="how many new tokens to generate")
parser.add_argument('--temperature', type=float, default=0.8,
                    help="below 1.0 is more coherent and repetitive, above is more varied")
parser.add_argument('--top-k', type=int, default=0,
                    help="sample only from the k most likely tokens (0 disables)")
parser.add_argument('--top-p', type=float, default=0.9,
                    help="nucleus sampling: keep the smallest token set summing to this probability")
parser.add_argument('--repetition-penalty', type=float, default=1.15,
                    help="above 1.0 discourages reusing tokens already in the context")
parser.add_argument('--samples', type=int, default=1,
                    help="how many independent continuations to produce")
parser.add_argument('--weights', default='transformer_weights.pth')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = tiktoken.get_encoding("gpt2")

model = Transformer(vocab_size=tokenizer.n_vocab, n_embd=n_embd, block_size=block_size,
                    num_blocks=num_blocks, n_head=n_head, head_size=head_size,
                    mlp_hidden=mlp_hidden)

state_dict = torch.load(args.weights, map_location='cpu')
state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

context_ids = tokenizer.encode_ordinary(args.prompt)
context = torch.tensor([context_ids], dtype=torch.long, device=device)

for i in range(args.samples):
    generated = model.generate(
        context,
        max_new_tokens=args.tokens,
        block_size=block_size,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Decode only the newly generated tail, so the prompt isn't reprinted as
    # though the model produced it.
    continuation = tokenizer.decode(generated[0].tolist()[len(context_ids):])

    if args.samples > 1:
        print(f"--- sample {i + 1} ---")
    print(args.prompt, end='')
    print(continuation)
    print()
