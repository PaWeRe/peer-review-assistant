# Peer Review Assistant

Local, privacy-preserving peer-review assistant powered by [Ollama](https://ollama.com). No data leaves your machine.

## Setup

```bash
uv venv && uv pip install -r requirements.txt
ollama serve
ollama pull qwen3-vl:30b
```

## TMI Review (1st Round, Vision)

2-pass pipeline: PDF pages are rendered as images so the model sees figures, tables, and equations directly.

```bash
uv run python tmi_review_assistant.py \
  -m tmi/paper.pdf \
  -o tmi/review.md --full-report
```

With optional reviewer notes to guide the analysis:

```bash
uv run python tmi_review_assistant.py \
  -m tmi/paper.pdf \
  -n tmi/notes.txt \
  -o tmi/review.md --full-report
```

## MEDIA Review (2nd Round)

3-pass text pipeline comparing original draft, reviewer comments, and revised draft.

```bash
uv run python media_review_assistant.py \
  --draft1 media/first_draft.pdf \
  --comments media/reviewer_comments.txt \
  --draft2 media/revised_draft.pdf \
  -o media/review.md --full-report
```

## General-Purpose Reviewer

```bash
uv run python review_assistant.py manuscript.pdf
```

## Options

All scripts support `--help`. Common flags:

| Flag | Description |
|------|-------------|
| `--model` | Ollama model name (default varies per script) |
| `--context-length` | Context window size in tokens (default 65536) |
| `--dpi` | PDF render resolution (TMI script only, default 150) |
| `--notes`, `-n` | Reviewer notes file to guide analysis (TMI script only) |
| `--full-report` | Save intermediate analysis alongside final review |
| `--no-stream` | Wait for full response instead of streaming |
| `--list-models` | List available Ollama models and exit |

## Performance & Context Tuning

Vision-language models convert each PDF page image into visual tokens. At 150 DPI
a typical page produces ~6k tokens; a 13-page paper consumes ~25-30k image tokens
before the text prompt and model output. This makes `--context-length` and `--dpi`
the two most important knobs.

**Context length** controls how much total content (images + prompt + output) the
model can handle. If the Pass 1 analysis is truncated (missing sections), increase
it. The tradeoff is prompt-evaluation time: attention scales roughly O(n²) with
sequence length, so doubling context ~quadruples prefill time.

**DPI** controls image resolution and therefore image token count. Lower DPI =
fewer tokens = faster prefill and more room for output. 150 DPI is a good default;
figures and text remain readable. Drop to 120 if you need speed; go to 200 for
maximum visual detail at the cost of slower runs.

**MoE models** (e.g. `qwen3-vl:30b` which is ~3B active params of 31B total) have
a modest KV cache relative to their size. All expert weights must be in memory
(~23 GB for qwen3-vl:30b Q4_K_M), but the KV cache grows slowly with context.
Token generation (decode) is fast (~50 tok/s) regardless of context size.

Reference timings on Apple M2 Max (32 GB), 13-page paper at 150 DPI:

| Context | Prefill | Decode | Total |
|---------|---------|--------|-------|
| 32k | ~6 min | ~1 min | ~8 min (but Pass 1 may truncate) |
| 65k | ~80 min | ~2 min | ~83 min (full Pass 1) |
