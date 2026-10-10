# Web UI

A single-page chat-style interface for the trained model. Standard library
only — no Flask, no Gradio. The container already has `torch` and `tiktoken`,
which is all this needs.

## Running

From inside the container, at the repo root:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 ui/server.py
```

It loads `transformer_weights.pth` (about 10 seconds) and then serves on port
8000.

## Reaching it from the browser

The training container was created without a published port, so
`localhost:8000` on the host will **not** work. Two options:

**Use the container's bridge IP** (works immediately, nothing to restart):

```bash
docker inspect rocm-training --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
```

Then open `http://<that-ip>:8000` — typically `http://172.17.0.2:8000`. Note
the IP can change if the container is ever recreated.

**Publish the port properly** (survives recreation, needs a new container):
add `-p 8000:8000` to the `docker run` command, then use
`http://localhost:8000`.

## The two modes

This is a **base** language model. It was trained to continue text and has had
no instruction tuning, so it is not a chatbot.

- **Continue text** — your input is fed in verbatim and extended. This is what
  the model natively does and what it is best at. Give it the opening of an
  explanatory sentence: *"Photosynthesis is the process by which"*.
- **Q&A (few-shot)** — your input is wrapped in a question/answer pattern with
  two worked examples, which coaxes answer-shaped output and gives generation
  a reliable place to stop. Works often enough to demo, but it is imitating a
  pattern, not answering a question.

## Sliders

| Control | Default | What it does |
|---|---|---|
| Temperature | 0.80 | Below 1.0 is more coherent and more repetitive; above is more varied and less so |
| Top-k | off | Keeps the k likeliest tokens. A fixed cutoff regardless of the model's confidence |
| Top-p | 0.90 | Keeps the smallest token set summing to p. Adapts to confidence; better than top-k here |
| Repetition penalty | 1.15 | Raise to break verbatim loops. Too high and grammar suffers, since common words get penalised too |
| Sentences | 2 | Generation stops on a sentence boundary, so output never breaks mid-clause |
| Token ceiling | 80 | Hard cap, used only if the sentence limit isn't reached first |

Sentence detection requires a terminator followed by whitespace, so decimals
like `3.14` don't end a sentence. If the token ceiling is hit before the
sentence limit, whatever was generated is flushed as-is.

If output loops, raise the repetition penalty toward 1.3 or temperature toward
0.95. If it turns incoherent, lower both.

## Expected behaviour

Fluent, grammatical, topically consistent prose that confidently invents
names, dates and facts. A representative sample:

> Tokyo is the capital of Japan and is located on the island of Hokusai in the
> far north of Japan (modern day Korea).

Correct core fact, invented specifics. That is what 153M parameters and 1.6B
training tokens buys.

## Notes

- Generation is serialised behind a lock. The model is one shared resource on
  one GPU, and VRAM is tight enough that concurrent requests could OOM.
- Don't run this while training is running — they will compete for VRAM.
- Tokens stream to the browser as they are produced, so output appears
  progressively rather than after a multi-second wait.
