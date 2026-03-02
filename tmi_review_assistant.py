#!/usr/bin/env python3
"""
TMI peer-review assistant using Ollama vision models.

2-pass pipeline:
  Pass 1 — Detailed analysis (vision: PDF page images + reviewer notes)
  Pass 2 — Formal TMI review (evaluation matrix, comments, recommendation)

Renders PDF pages as images so the model sees figures, tables, and equations
exactly as a human reviewer would.
All processing stays on your machine — no data is sent externally.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

_print = print


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:30b"
DEFAULT_DPI = 150
# Context length trades off completeness vs. prompt-eval time. Key factors:
#   - VL models convert each page image into visual tokens (~(W/28)*(H/28) per
#     page). At 150 DPI a US-letter page is ~1913x2475 px ≈ 6k tokens. With 13
#     pages that's ~25-30k image tokens alone, before the text prompt and output.
#   - MoE models (like qwen3-vl:30b, ~3B active of 31B total) have a modest KV
#     cache relative to their total param count, so larger contexts are feasible
#     memory-wise. However, attention during prompt evaluation scales ~O(n²) with
#     sequence length, so doubling context roughly quadruples prefill time.
#   - On an M2 Max (32 GB): 32k ctx → ~6 min prefill, 65k → ~80 min prefill.
#     Token generation (decode) is fast regardless (~50 tok/s) since it's O(n).
#   - If Pass 1 output is truncated (missing sections 2-5), increase this value.
#     If prefill is too slow, lower DPI first (reduces image tokens without
#     affecting context budget).
DEFAULT_CONTEXT_LENGTH = 65536

# ---------------------------------------------------------------------------
# TMI evaluation criteria
# ---------------------------------------------------------------------------

TMI_EVAL_CATEGORIES = [
    "Scientific Merit",
    "Originality",
    "Impact",
    "Practical Value",
    "Reader Interest",
    "Overall Evaluation",
]

TMI_RATING_SCALE = "Excellent / Very Good / Good / Fair / Poor"

TMI_RECOMMENDATIONS = [
    "Accept as submitted",
    "Accept after minor revision (no re-review needed)",
    "Major revision required (re-review needed)",
    "Reject and encourage resubmission",
    "Reject — no further consideration",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NOTES_BLOCK = """\

The reviewer has provided their working notes below. These are a mix of \
observations, open questions, and preliminary impressions — NOT established \
conclusions. Some may reflect incomplete reading of the manuscript.

For each point in the notes:
- Independently verify whether the concern is warranted by checking the \
  manuscript pages directly.
- If the note is phrased as a question, investigate it and report what you \
  find — do not assume the answer.
- Do NOT amplify tentative observations into definitive criticisms.

REVIEWER NOTES:
---
{notes}
---
"""

PASS1_PROMPT_TEMPLATE = """\
You are helping a peer reviewer assess a manuscript submitted to \
IEEE Transactions on Medical Imaging (TMI). The complete paper is provided \
as page images. Your job is to perform an EXHAUSTIVE, paragraph-by-paragraph \
and figure-by-figure analysis — as if you were the most meticulous reviewer \
in the field.
{notes_block}\
IMPORTANT GUIDELINES:

1. SYMBOL CAUTION: When reading mathematical notation from page images, be \
careful with visually similar symbols. In particular, ∝ (proportional to) \
can look like × or other operators; ∈ can resemble ε; ≈ can look like =. \
If you are uncertain about a symbol, describe what you observe rather than \
asserting an error. Do NOT claim notation errors unless you are confident \
about what the symbol actually is.

2. EPISTEMIC HONESTY: Clearly distinguish between:
   (a) Definite issues you can verify from the manuscript (e.g., missing \
       standard deviations in a table),
   (b) Potential concerns that warrant author clarification (e.g., "the \
       rationale for X is not explained — it may exist but I could not \
       find it"), and
   (c) Suggestions for improvement (e.g., "adding Y would strengthen Z").
   Do not present (b) or (c) as definitive flaws.

3. INTERNAL CONSISTENCY: If you acknowledge that a specific combination or \
application is new, do not simultaneously dismiss it as incremental without \
explaining the distinction. Novelty can exist at the combination or \
application level even when individual components have precedent.

4. DEEP FIGURE & TABLE ANALYSIS: For every figure and table, examine them \
as a domain expert would:
   - Figures: study each panel, axis labels, legends, color maps, scale \
     bars, visual artifacts, and whether the visual evidence genuinely \
     supports the claims made in the text. Check for cherry-picked examples, \
     and whether qualitative differences between methods are actually \
     discernible or marginal.
   - Tables: verify column headers, units, metric definitions, whether \
     bold/underline conventions are applied consistently, whether standard \
     deviations/confidence intervals/p-values are present, and whether \
     reported gains are statistically meaningful or within noise.

5. STATE-OF-THE-ART CONTEXTUALIZATION: You have extensive knowledge of the \
medical imaging and computer vision literature encoded in your weights. \
USE IT ACTIVELY. For the specific subfield this paper targets:
   - Identify the current SOTA methods (published up to your knowledge \
     cutoff) and assess whether the authors compare against them.
   - Flag if the authors benchmark only against older or weaker baselines \
     while ignoring stronger recent methods.
   - Evaluate whether the claimed contributions are genuinely novel given \
     the current state of the field, or whether very similar ideas have \
     appeared in recent work the authors may have missed.
   - Identify key methodological trends in this subfield that the paper \
     should acknowledge or build upon.

Produce a structured analysis covering the sections below. Be concrete: \
cite page numbers, figure/table numbers, equation numbers, and quote or \
closely paraphrase specific author claims. Avoid generic praise or filler. \
Do NOT produce a recommendation or rating — the formal review will be \
generated in a separate step.

## 1. Paper Digest

5-6 sentences: problem addressed, proposed method (core idea, not a laundry \
list of components), datasets used, headline quantitative results.

Then provide one sentence for EVERY figure and table in the manuscript \
(not just the ones you consider most important). Enumerate them all \
(Fig. 1, Fig. 2, ..., Table I, Table II, ...).

## 2. Scientific Merit

Evaluate the soundness and rigor of the methodology paragraph by paragraph. \
For each major section of the paper, identify claims and check whether they \
are supported by evidence within the manuscript. For each concern, indicate \
whether it is a verified flaw vs. something that may be addressed in the \
paper but you could not locate. Comment on experimental design, \
hyperparameter choices (are they justified or arbitrary?), and \
reproducibility (code availability, implementation details, compute \
requirements, runtime/memory analysis).

## 3. Novelty Audit

For EACH novelty claim the authors make:
- Quote or paraphrase the claim with section/page reference.
- Identify the closest prior work and how the authors differentiate.
- Assess whether the differentiation is substantiated or superficial.
- Draw on your knowledge of the broader field: has a very similar approach \
  been published elsewhere that the authors may not have cited?

Be precise about SCOPE: if the authors claim novelty for a specific \
combination (e.g., "first to apply X to Y"), evaluate that specific claim, \
not the broader components in isolation. Acknowledge genuine novelty at \
the application or combination level when warranted.

## 4. Literature Gaps

This section is CRITICAL. Draw on your full knowledge of the field to \
identify references that are NOT already cited by the authors but SHOULD be. \
Repeating the paper's own citations is not useful — the reviewer can see those.

For each missing reference:
- Give enough detail to verify (first author, short title, venue, year).
- Explain concretely WHY it matters: does it present a competing method, \
  an overlapping technique, a relevant benchmark, or a theoretical \
  framework the authors should discuss?
- Mark confidence: [HIGH] if you are certain this is a real, published \
  work; [MEDIUM] if you are fairly sure but details may be slightly off; \
  [LOW] if uncertain.

Also flag any works that ARE cited but not adequately discussed — e.g., a \
key baseline that is cited by number but never analyzed for its limitations \
relative to the proposed method.

Do NOT fabricate bibliographic details. If you cannot identify genuinely \
missing references with reasonable confidence, say so honestly rather than \
padding the list. But DO try hard — this is one of the most valuable things \
a reviewer can provide.

## 5. Experimental Scrutiny

Address each explicitly:
- Are baselines current SOTA or outdated? Are they fairly compared \
  (same backbone, data splits, training budget, hardware)?
- Are datasets standard for this subfield and sufficient for the claimed \
  scope of generalizability?
- Do effect sizes support the claims, or are improvements marginal / \
  within noise / lacking significance tests?
- Are ablations sufficient to isolate EACH claimed contribution?
- Is there runtime/memory/compute analysis? If not, flag this.
- Are there failure case analyses? If not, flag this.
- Is there statistical rigor (std, CI, p-values, sample sizes)?

## 6. Figures & Tables — Deep Analysis

Cover EVERY figure and table in the manuscript. For each one:
- What does it aim to convey?
- Study it closely: are axis labels correct, are differences between \
  methods actually visible in qualitative comparisons, do the numbers \
  in tables align with claims in the text?
- Is it effective at supporting the paper's narrative, or is it \
  misleading, redundant, or poorly designed?
- Would a different visualization or metric be more informative?
- Flag any visual evidence that contradicts or only weakly supports \
  the text claims (e.g., qualitative comparisons where methods look \
  nearly identical despite claimed improvements).

## 7. Writing & Presentation

Comment on structure, clarity, grammar, and whether technical details are \
accessible. Note any repetitive or unclear passages.

## 8. Specific Issues

Numbered list of concrete, actionable problems the authors should address. \
Reference specific sections, figures, equations, or tables. Only include \
issues you have verified — do not include speculative concerns here.
"""

PASS2_PROMPT_TEMPLATE = """\
You are writing the FORMAL peer review for a manuscript submitted to \
IEEE Transactions on Medical Imaging (TMI). You have already performed \
a detailed analysis of the paper (provided below). Now synthesize that \
analysis into the official review format.
{notes_block}\
DETAILED ANALYSIS:
---
{pass1_output}
---

IMPORTANT PRINCIPLES:
- Base ratings on VERIFIED issues, not speculative ones. A concern flagged \
  as "needs clarification" should weigh less than a confirmed flaw.
- Acknowledge genuine strengths with the same specificity as weaknesses. \
  A balanced review is more credible and useful than an entirely critical one.
- If the detailed analysis contains contradictions (e.g., acknowledging \
  novelty while calling it incremental), resolve them — take a clear position.
- For missing references, only include works NOT already cited by the authors.
- Be CONCISE. Reviewers and authors value brevity and precision over length. \
  Every sentence should carry information — no filler, no restating of the \
  obvious, no generic praise.

Produce your review in EXACTLY the following format:

## 1. Evaluation Matrix

Rate each category using the scale: {rating_scale}

| Category | Rating |
|----------|--------|
{eval_rows}

Immediately below the table, provide a 1-2 sentence justification for EACH \
category rating.

## 2. Comments to the Author

This section must be CONCISE and DIRECT — aim for quality over quantity. \
Follow this exact structure:

**Significance and innovation:** One focused paragraph (3-5 sentences max). \
Summarize the core contribution, credit what is genuinely new, note the \
paper's strengths (clear figures, good ablations, etc.), and briefly \
position the work relative to SOTA. If novelty is at the combination or \
application level, say so clearly. Do NOT rehash the abstract.

**Specific concerns:**
A numbered list of the most important issues. Each item should:
- Start with a bolded category tag (e.g., **Methodology**, **Statistical \
  validation**, **Reproducibility**, **Benchmarking/Evaluation**, \
  **Presentation**, **Scope/Claims**).
- Include specific page, figure, table, or equation references in parentheses.
- Be 2-4 sentences: state the issue, why it matters, and what the authors \
  should do about it.
- Consolidate related issues into a single numbered item rather than \
  listing many small points.
Aim for 3-6 items covering the most impactful concerns. Do NOT include \
minor nitpicks or speculative issues.

**References:**
List specific references NOT already cited in the paper that the authors \
should add or engage with. For each:
- Provide full citation detail (authors, year, short title, venue).
- Explain in one sentence why it is relevant.
- If a reference IS cited but insufficiently discussed (e.g., cited as a \
  number but never analyzed or contrasted), note this separately with an \
  italicized remark.
If no genuinely missing references were identified, state that honestly.

## 3. Recommendation

Choose EXACTLY ONE of:
{recommendations}

Format: **Recommendation: <chosen option>**

Justify in 2-3 sentences referencing the key strengths and the most \
critical weaknesses. Be direct.
"""


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_pages(filepath: str, dpi: int = DEFAULT_DPI) -> list[str]:
    """Render each PDF page to a base64-encoded PNG with progress tracking."""
    import fitz

    doc = fitz.open(filepath)
    n_pages = len(doc)
    pages = []

    try:
        from tqdm import tqdm
        iterator = tqdm(doc, total=n_pages, desc="  Rendering pages", unit="pg")
    except ImportError:
        iterator = doc

    first = True
    for page in iterator:
        pix = page.get_pixmap(dpi=dpi)
        if first:
            print(f"  Page dimensions: {pix.width}x{pix.height} px")
            first = False
        png_bytes = pix.tobytes("png")
        if len(png_bytes) < 100:
            print(f"  Warning: page {page.number + 1} rendered to only "
                  f"{len(png_bytes)} bytes — may be blank")
        pages.append(base64.b64encode(png_bytes).decode("ascii"))

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def query_ollama(
    prompt: str,
    model: str,
    images: list[str] | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    stream: bool = True,
) -> str:
    """Send prompt (+ optional page images) to a local Ollama model."""
    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    payload = {
        "model": model,
        "messages": [message],
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_ctx": context_length,
        },
    }

    try:
        resp = requests.post(
            OLLAMA_CHAT_URL, json=payload, timeout=7200, stream=stream,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        sys.exit(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve"
        )
    except requests.Timeout:
        sys.exit("Ollama request timed out (2-hour limit).")
    except requests.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        sys.exit(f"Ollama HTTP error: {e}\n{body}")
    except Exception as e:
        sys.exit(f"Ollama error: {e}")

    if not stream:
        data = resp.json()
        return data.get("message", {}).get("content", "")

    chunks: list[str] = []
    token_count = 0
    t0 = time.time()
    eval_count = 0

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            _print(token, end="", flush=True)
            chunks.append(token)
            token_count += 1
        if data.get("done"):
            eval_count = data.get("eval_count", token_count)
            break

    elapsed = time.time() - t0
    _print()
    tok_s = eval_count / elapsed if elapsed > 0 else 0
    print(f"  [{eval_count} tokens in {elapsed:.0f}s — {tok_s:.1f} tok/s]")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_notes_block(notes_path: str | None) -> str:
    """Read reviewer notes and format for inclusion in prompts."""
    if not notes_path:
        return ""
    path = Path(notes_path)
    if not path.is_file():
        sys.exit(f"Notes file not found: {notes_path}")
    notes = path.read_text(encoding="utf-8").strip()
    if not notes:
        return ""
    return NOTES_BLOCK.format(notes=notes)


def run_pipeline(
    pages: list[str],
    notes_path: str | None,
    model: str,
    context_length: int,
    stream: bool,
) -> tuple[str, str]:
    """Run the 2-pass review pipeline. Returns (pass1, pass2) outputs."""
    notes_block = build_notes_block(notes_path)

    # --- Pass 1: Detailed analysis (vision) ---
    print("\n" + "=" * 60)
    print("[Pass 1/2] Detailed Analysis (vision model)")
    print("=" * 60 + "\n")

    prompt1 = PASS1_PROMPT_TEMPLATE.format(notes_block=notes_block)
    t0 = time.time()
    pass1 = query_ollama(
        prompt1, model,
        images=pages,
        context_length=context_length,
        stream=stream,
    )
    print(f"  Pass 1 total: {time.time() - t0:.0f}s")

    # --- Pass 2: Formal TMI review (text only) ---
    print("\n" + "=" * 60)
    print("[Pass 2/2] Formal TMI Review")
    print("=" * 60 + "\n")

    eval_rows = "\n".join(f"| {cat} | |" for cat in TMI_EVAL_CATEGORIES)
    recommendations = "\n".join(f"- {r}" for r in TMI_RECOMMENDATIONS)

    prompt2 = PASS2_PROMPT_TEMPLATE.format(
        notes_block=notes_block,
        pass1_output=pass1,
        rating_scale=TMI_RATING_SCALE,
        eval_rows=eval_rows,
        recommendations=recommendations,
    )
    t0 = time.time()
    pass2 = query_ollama(
        prompt2, model,
        context_length=context_length,
        stream=stream,
    )
    print(f"  Pass 2 total: {time.time() - t0:.0f}s")

    return pass1, pass2


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_review(pass2_output: str) -> str:
    border = "=" * 70
    return "\n".join([
        border,
        "IEEE TRANSACTIONS ON MEDICAL IMAGING — PEER REVIEW",
        border,
        "",
        pass2_output.strip(),
        "",
        border,
        "Generated by local Ollama review assistant. No data left this machine.",
        border,
    ])


def format_full_report(pass1: str, pass2: str) -> str:
    border = "=" * 70
    return "\n".join([
        border,
        "IEEE TRANSACTIONS ON MEDICAL IMAGING — PEER REVIEW  (full report)",
        border,
        "",
        "## Pass 1: Detailed Analysis",
        "-" * 40,
        pass1.strip(),
        "",
        "## Pass 2: Formal Review",
        "-" * 40,
        pass2.strip(),
        "",
        border,
        "Generated by local Ollama review assistant. No data left this machine.",
        border,
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "TMI review assistant — 2-pass local Ollama VL pipeline. "
            "Sends PDF pages as images so figures/tables are visible. "
            "Privacy-preserving: all data stays on this machine."
        ),
    )
    parser.add_argument(
        "--manuscript", "-m", required=True,
        help="Path to the manuscript PDF",
    )
    parser.add_argument(
        "--notes", "-n",
        help="Path to reviewer notes file (TXT/MD) to guide the analysis",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save review to this markdown file",
    )
    parser.add_argument(
        "--full-report", action="store_true",
        help="Also save full report with intermediate Pass 1 analysis",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama VL model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"PDF render resolution in DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
        help=f"Model context window (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Wait for full response instead of streaming tokens",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available Ollama models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = resp.json().get("models", [])
            _print("Available Ollama models:")
            for m in models:
                size_gb = m.get("size", 0) / 1e9
                _print(f"  {m['name']:<40s} {size_gb:.1f} GB")
        except Exception:
            _print("Cannot connect to Ollama. Is it running?")
        return

    if not os.path.isfile(args.manuscript):
        sys.exit(f"File not found: {args.manuscript}")

    ext = Path(args.manuscript).suffix.lower()
    if ext != ".pdf":
        sys.exit(f"Only PDF files are supported (got {ext}).")

    # Render PDF pages as images
    print(f"Rendering {args.manuscript} at {args.dpi} DPI ...")
    pages = render_pdf_pages(args.manuscript, dpi=args.dpi)
    raw_mb = sum(len(p) for p in pages) * 3 / 4 / 1e6
    print(f"  {len(pages)} pages (~{raw_mb:.1f} MB image data)")

    if args.notes:
        notes_text = Path(args.notes).read_text(encoding="utf-8").strip()
        print(f"  Reviewer notes: {args.notes} ({len(notes_text):,} chars)")

    print(f"\nModel:   {args.model}")
    print(f"Context: {args.context_length:,} tokens")
    print("All processing is LOCAL. No data leaves this machine.")

    t_start = time.time()
    pass1, pass2 = run_pipeline(
        pages,
        notes_path=args.notes,
        model=args.model,
        context_length=args.context_length,
        stream=not args.no_stream,
    )
    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_elapsed:.0f}s "
          f"({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")

    if args.output:
        out = Path(args.output)
        out.write_text(format_review(pass2), encoding="utf-8")
        print(f"Review saved to: {out}")

        if args.full_report:
            full_path = out.with_stem(out.stem + "_full")
            full_path.write_text(
                format_full_report(pass1, pass2), encoding="utf-8",
            )
            print(f"Full report saved to: {full_path}")


if __name__ == "__main__":
    main()
