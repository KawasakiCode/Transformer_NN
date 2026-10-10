"""Web UI for the trained model.

Serves a single page on http://localhost:8000 where you type a prompt and the
model streams a continuation back, with sliders for the sampling parameters.

Deliberately uses only the standard library plus torch/tiktoken, which the
training container already has - no Flask, no Gradio, nothing new to install.

Run it from inside the container:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 ui/server.py

Note this is a BASE language model with no instruction tuning. It continues
text rather than answering it. The UI offers two modes:
  - Continue: your text is fed in verbatim and extended. What the model
    natively does, and what it is best at.
  - Q&A: your text is wrapped in a few-shot question/answer pattern to coax
    answer-shaped output. Often works, frequently doesn't, and is not the
    same thing as a chat model.
"""

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import Transformer, strip_obsolete_keys  # noqa: E402

# Must match main.py for the run that produced the weights file.
BLOCK_SIZE = 1024
N_EMBD = 512
N_HEAD = 16
HEAD_SIZE = 32
NUM_BLOCKS = 24
MLP_HIDDEN = 3072

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(os.path.dirname(HERE), 'transformer_weights.batched.pth')
PORT = int(os.environ.get('PORT', 8000))

# The model is a single shared resource on one GPU, and VRAM here is tight
# enough that two concurrent generations could OOM. Serialise them.
GPU_LOCK = threading.Lock()

# Few-shot preamble for Q&A mode. Gives the model a pattern to imitate, and a
# reliable stop marker ("\nQ:") for when its answer has run its course.
QA_PREAMBLE = (
    "Q: What is the capital of France?\n"
    "A: Paris is the capital of France.\n\n"
    "Q: What is photosynthesis?\n"
    "A: Photosynthesis is the process by which plants convert sunlight into "
    "chemical energy, producing oxygen as a by-product.\n\n"
)

print('Loading tokenizer...')
tokenizer = tiktoken.get_encoding('gpt2')

print(f'Building model and loading {WEIGHTS}...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(vocab_size=tokenizer.n_vocab, n_embd=N_EMBD, block_size=BLOCK_SIZE,
                    num_blocks=NUM_BLOCKS, n_head=N_HEAD, head_size=HEAD_SIZE,
                    mlp_hidden=MLP_HIDDEN)
state = torch.load(WEIGHTS, map_location='cpu')
state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
state = strip_obsolete_keys(state)
model.load_state_dict(state)
del state
model = model.to(device).eval()
print(f'Model ready on {device}.')


# A sentence terminator followed by whitespace. The lookahead requires the
# whitespace to actually be present rather than matching end-of-buffer, so
# "3.14" or a mid-word "etc." does not end a sentence just because generation
# happens to pause there - we wait to see what comes next.
SENTENCE_END = re.compile(r'[.!?]["\')\]]?(?=\s)')


def sentence_cut(text, limit):
    """Index just past the limit-th complete sentence, or None if not reached."""
    if not limit:
        return None
    for count, match in enumerate(SENTENCE_END.finditer(text), start=1):
        if count >= limit:
            return match.end()
    return None


def build_prompt(text, mode):
    """Return (text_fed_to_model, stop_strings)."""
    if mode == 'qa':
        return QA_PREAMBLE + f"Q: {text.strip()}\nA:", ["\nQ:", "\n\n"]
    return text, []


def stream_tokens(text, mode, settings):
    """Yield chunks of generated text, stopping at a stop string if one appears."""
    prompt, stops = build_prompt(text, mode)
    ids = tokenizer.encode_ordinary(prompt)

    # Leave room for the completion inside the context window.
    max_new = min(int(settings.get('tokens', 200)), 512)
    ids = ids[-(BLOCK_SIZE - max_new):]
    context = torch.tensor([ids], dtype=torch.long, device=device)

    top_k = int(settings.get('top_k', 0)) or None
    top_p = float(settings.get('top_p', 0.9))

    max_sentences = int(settings.get('sentences', 2))

    produced_ids = []
    emitted = 0
    # Never emit the last few characters while they could still turn out to be
    # the opening of a stop string: "\nQ" has to be withheld until we know
    # whether the next token makes it "\nQ:" (stop) or "\nQuite" (keep). Once
    # text has been streamed it cannot be taken back. At least one character
    # is always withheld so sentence detection can see what follows a period.
    holdback = max((len(s) for s in stops), default=1) - 1
    if max_sentences:
        holdback = max(holdback, 1)

    with GPU_LOCK:
        for token in model.generate_stream(
            context,
            max_new_tokens=max_new,
            block_size=BLOCK_SIZE,
            temperature=max(float(settings.get('temperature', 0.8)), 0.01),
            top_k=top_k,
            top_p=top_p if top_p < 1.0 else None,
            repetition_penalty=float(settings.get('repetition_penalty', 1.15)),
        ):
            produced_ids.append(token)

            # Decode the whole completion each step rather than token by
            # token: a single BPE token is often half a character in UTF-8,
            # and decoding it alone produces replacement characters.
            whole = tokenizer.decode(produced_ids)

            hit = next((s for s in stops if s in whole), None)
            if hit:
                yield whole[:whole.index(hit)][emitted:]
                return

            # Stop on a sentence boundary rather than mid-clause, which is what
            # a token budget alone gives you.
            cut = sentence_cut(whole, max_sentences)
            if cut is not None:
                yield whole[:cut][emitted:]
                return

            safe = len(whole) - holdback
            if safe > emitted:
                yield whole[emitted:safe]
                emitted = safe

        # Ran out of tokens rather than hitting a stop - flush whatever was
        # being withheld.
        whole = tokenizer.decode(produced_ids)
        if len(whole) > emitted:
            yield whole[emitted:]


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *a):  # quieter console
        pass

    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_error(404)
            return
        with open(os.path.join(HERE, 'index.html'), 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/generate':
            self.send_error(404)
            return

        length = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError:
            self.send_error(400, 'bad json')
            return

        text = (payload.get('text') or '').strip()
        if not text:
            self.send_error(400, 'empty prompt')
            return

        # Newline-delimited JSON, flushed per chunk, so the browser can render
        # text as it arrives instead of waiting for the full completion.
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

        def write_chunk(obj):
            data = (json.dumps(obj) + '\n').encode('utf-8')
            self.wfile.write(f'{len(data):X}\r\n'.encode('ascii') + data + b'\r\n')
            self.wfile.flush()

        try:
            for piece in stream_tokens(text, payload.get('mode', 'continue'), payload):
                if piece:
                    write_chunk({'text': piece})
            write_chunk({'done': True})
        except (BrokenPipeError, ConnectionResetError):
            return  # viewer navigated away mid-generation
        except Exception as e:                      # keep the server alive
            try:
                write_chunk({'error': str(e)})
            except Exception:
                pass
        finally:
            try:
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except Exception:
                pass


if __name__ == '__main__':
    print(f'\n  http://localhost:{PORT}\n')
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
