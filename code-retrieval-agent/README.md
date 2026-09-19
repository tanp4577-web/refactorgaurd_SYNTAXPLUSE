# Code Retrieval Agent

Built for the **Samsung PRISM Gen AI Hackathon 3.0** — Theme: **Agentic Code
Intelligence** — Team **SyntaxPulse**.

## Problem statement (in our own words)

Given a natural-language query and a large codebase, return the code
snippets most relevant to that query, ranked by relevance. This is the code
retrieval problem the official theme guideline describes: a developer (or an
AI agent) asking "how is the input validated before it reaches the main
function?" should get the actually-relevant snippet ranked first, out of
potentially thousands of candidates that will never all fit in an LLM's
context window.

## Solution

A CPU-friendly embedding-based retrieval pipeline:

```
Query ──▶ preprocess_query ──▶ embed (CodeRetrievalEncoder) ──┐
                                                                 ├──▶ cosine similarity ──▶ ranked snippets
Codebase ──▶ preprocess_snippet ──▶ embed (CodeRetrievalEncoder) ──┘
```

- **`src/encoder.py`** — wraps a pretrained code-aware sentence-transformers
  model (default: `jinaai/jina-embeddings-v2-base-code`) behind MTEB's real
  `AbsEncoder` interface, so it plugs directly into MTEB's official
  evaluation harness.
- **`src/indexer.py`** — a version-aware index (`CodeIndex`) that caches
  embeddings by content hash, so re-indexing a changed codebase only
  re-embeds snippets that actually changed — this is the P1 "retrieval
  across versions" requirement from the guideline, addressed directly rather
  than by re-embedding everything on every change.
- **`src/run_eval.py`** — runs the official required evaluation: MTEB's
  `AppsRetrieval` task (CoIR-Retrieval/apps dataset), producing the
  NDCG@10/MRR JSON file required for submission screening.
- **`src/demo.py`** — a small runnable demo using the guideline's own
  example snippets, showing real ranked results end to end, plus a live
  demonstration of the incremental re-indexing behavior.

## Setup

```bash
git clone <this-repo-url>
cd code-retrieval-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running the required evaluation (produces the submission JSON)

```bash
python -m src.run_eval
```

This downloads the CoIR-Retrieval/apps dataset and the embedding model from
Hugging Face on first run (requires internet access), runs MTEB's official
`AppsRetrieval` task, and writes `appsretrieval_results.json` — the file
required for the GitHub release per the official submission instructions
(tag: `PRISM_GENAI_HACKATHON_Y2026`).

## Running the demo (for the video)

```bash
python -m src.demo
```

Shows ranked retrieval results for natural-language queries against a small
set of code snippets, plus the incremental re-indexing behavior.

## Running the unit tests

```bash
python3 tests/test_encoder.py
python3 tests/test_indexer.py
```

These test the encoder's and indexer's internal logic using injected fake
models — they run offline, with no model downloads or network access
required, and are the tests we could verify in a network-restricted
environment before handing this off. The real end-to-end evaluation
(`run_eval.py`) still needs to be run once with real internet access to
confirm the actual NDCG@10/MRR numbers.

## Project structure

```
code-retrieval-agent/
├── src/
│   ├── encoder.py    — MTEB-compatible embedding encoder (P0)
│   ├── indexer.py     — version-aware incremental index (P1)
│   ├── run_eval.py    — official MTEB evaluation → submission JSON
│   └── demo.py         — runnable demo for the video
├── tests/
│   ├── test_encoder.py
│   └── test_indexer.py
└── requirements.txt
```

## Known limitations / honest scope notes

- `preprocess_query` and `preprocess_snippet` are currently identity
  functions (just whitespace trimming) — clear seams left for further
  improvement (query rewriting/categorization, snippet cleaning,
  multi-pass retrieval / reranking) but not yet built out, given time
  constraints.
- The Bonus goal ("Evolutionary Retrieval" — retrieving across *all*
  historical versions simultaneously) is not implemented; `CodeIndex`
  supports fast re-indexing to the *current* version only.
- The actual NDCG@10/MRR numbers from `run_eval.py` have not yet been
  recorded in this README — run it and update this section with the real
  scores before final submission.

## Team SyntaxPulse

Samsung PRISM Gen AI Hackathon 3.0 — Agentic Code Intelligence theme.
