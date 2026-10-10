"""Convert a per-head checkpoint to the batched-attention layout.

attention.py used to register one Head module per attention head, each with
its own query/key/value matrix:

    blocks.0.sa.heads.0.query.weight    (32, 512)
    blocks.0.sa.heads.1.query.weight    (32, 512)
    ...  16 of these per projection per layer

It now uses a single matrix per projection covering all heads:

    blocks.0.sa.query.weight            (512, 512)

These hold the same numbers. nn.Linear weight is (out_features, in_features),
so stacking the 16 per-head matrices along dim 0 produces exactly the batched
matrix, and the view(B, T, heads, head_size) in the new forward splits the
output back into the same 16 contiguous groups in the same order. proj is
untouched - it already consumed that same concatenation.

So this is a lossless reshuffle, not a retrain. The script proves it rather
than asserting it: --verify runs both the old and new computation on identical
random input and compares the outputs.

Only model weights are converted. Optimizer state is not, because Adam keys
its moment buffers by parameter index and the index list changes when 1152
tensors become 72 - which matters only for resuming training, and v1 is
finished. Convert transformer_weights.pth to keep generate.py and the UI
working with the trained model.

Usage:
    python3 convert_checkpoint.py transformer_weights.pth
    python3 convert_checkpoint.py transformer_weights.pth -o v1_batched.pth
    python3 convert_checkpoint.py transformer_weights.pth --verify
"""

import argparse
import collections
import re
import sys

import torch
from torch.nn import functional as F

HEAD_KEY = re.compile(r'^(?P<prefix>.*\.sa)\.heads\.(?P<head>\d+)\.(?P<proj>query|key|value)\.weight$')


def convert(state_dict):
    """Return (new_state_dict, number_of_layers_converted)."""
    # prefix -> projection -> {head index: tensor}
    grouped = collections.defaultdict(lambda: collections.defaultdict(dict))
    passthrough = {}

    for key, value in state_dict.items():
        if key.endswith('.tril'):
            continue  # obsolete causal-mask buffer
        match = HEAD_KEY.match(key)
        if match:
            grouped[match['prefix']][match['proj']][int(match['head'])] = value
        else:
            passthrough[key] = value

    new = dict(passthrough)
    for prefix, projections in grouped.items():
        for proj, heads in projections.items():
            # Order by head index, not dict insertion order: state_dict keys
            # sort as strings, so head 10 would otherwise land before head 2.
            ordered = [heads[i] for i in sorted(heads)]
            new[f'{prefix}.{proj}.weight'] = torch.cat(ordered, dim=0)

    return new, len(grouped)


def verify(state_dict, new_state_dict, prefix, num_heads, head_size, n_embd, tol=1e-5):
    """Run the old per-head path and the new batched path on the same input."""
    torch.manual_seed(0)
    B, T = 2, 16
    x = torch.randn(B, T, n_embd, dtype=torch.float32)

    # --- old path: a head at a time, concatenated, exactly as it used to be
    outs = []
    for h in range(num_heads):
        q = x @ state_dict[f'{prefix}.heads.{h}.query.weight'].float().T
        k = x @ state_dict[f'{prefix}.heads.{h}.key.weight'].float().T
        v = x @ state_dict[f'{prefix}.heads.{h}.value.weight'].float().T
        outs.append(F.scaled_dot_product_attention(q, k, v, is_causal=True))
    old_out = torch.cat(outs, dim=-1)

    # --- new path: one projection per matrix, heads split by reshape
    def batched(name):
        w = new_state_dict[f'{prefix}.{name}.weight'].float()
        return (x @ w.T).view(B, T, num_heads, head_size).transpose(1, 2)

    new_out = F.scaled_dot_product_attention(batched('query'), batched('key'),
                                             batched('value'), is_causal=True)
    new_out = new_out.transpose(1, 2).contiguous().view(B, T, num_heads * head_size)

    delta = (old_out - new_out).abs().max().item()
    return delta, delta <= tol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('-o', '--out', help='defaults to <input>.batched.pth')
    parser.add_argument('--verify', action='store_true',
                        help='prove old and new attention agree numerically')
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--head-size', type=int, default=32)
    parser.add_argument('--n-embd', type=int, default=512)
    args = parser.parse_args()

    print(f'Loading {args.checkpoint}...')
    raw = torch.load(args.checkpoint, map_location='cpu')

    # Accept both a bare state_dict and a full training checkpoint.
    if isinstance(raw, dict) and 'model' in raw and 'optimizer' in raw:
        print('  (training checkpoint: converting the model weights only)')
        state = raw['model']
    else:
        state = raw

    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}

    if not any(HEAD_KEY.match(k) for k in state):
        print('Nothing to do - this checkpoint is already in the batched layout.')
        return 0

    new_state, layers = convert(state)

    # Compare excluding the tril buffers, which are deliberately dropped -
    # counting them in would make an intended 1.61GB saving look like data loss.
    dropped = sum(v.numel() for k, v in state.items() if k.endswith('.tril'))
    before = sum(v.numel() for v in state.values()) - dropped
    after = sum(v.numel() for v in new_state.values())
    print(f'  tensors: {len(state)} -> {len(new_state)}  ({layers} attention layers)')
    print(f'  weights: {before:,} -> {after:,}'
          f'   ({"identical" if before == after else "CHANGED - this is a bug"})')
    if dropped:
        print(f'  dropped: {dropped:,} values of obsolete causal masks '
              f'({dropped * 4 / 1e9:.2f} GB)')

    if args.verify:
        prefix = next(HEAD_KEY.match(k)['prefix'] for k in state if HEAD_KEY.match(k))
        delta, ok = verify(state, new_state, prefix, args.heads, args.head_size, args.n_embd)
        print(f'\nequivalence check on {prefix}:')
        print(f'  max |old - new| = {delta:.3e}   {"PASS" if ok else "FAIL"}')
        if not ok:
            print('  refusing to write a checkpoint that does not match')
            return 1

    out = args.out or args.checkpoint.replace('.pth', '') + '.batched.pth'
    torch.save(new_state, out)
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
