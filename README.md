# PaperRAG

PaperRAG is a local research tool for PDF papers. It parses and indexes them, then answers questions strictly from indexed excerpts, checking every claim against its original cited passage rather than a model's memory. Ingest, search, and previews are local and CPU-only.

- **Local-first ingest and search.** PyMuPDF extraction with a local Tesseract OCR fallback, CPU ONNX embeddings, and embedded LanceDB hybrid retrieval — no vector server, GPU, or JavaScript build.
- **Evidence-checked research.** Claims are verified in a separate call against their original cited excerpts; only supported, directly responsive claims become answer facts; hypotheses stay labeled as unproven.
- **One explicit endpoint.** Research needs a base URL, API key, and three model IDs you supply; secrets live in `.env`, provider options in TOML.
- **Recoverable storage.** Journal, chunk store, page cache, and model cache share one data directory behind a single-writer protocol.

## Quickstart

Prerequisites: a checkout of this repository, micromamba, and network access for initial dependencies, models, and (optionally) samples. All commands run from the checkout root.

```sh
micromamba create -y -f environment.yml
micromamba run -n paperrag python -m pip install -e .
micromamba run -n paperrag paperrag download-samples
micromamba run -n paperrag paperrag ingest papers
micromamba run -n paperrag paperrag search "What non-parametric memory does RAG use?" --json
micromamba run -n paperrag paperrag ui
```

Samples are optional: for your own PDFs, replace the download/ingest pair with `micromamba run -n paperrag paperrag ingest /path/to/pdfs`. Search returns indexed excerpts with source/page metadata; the UI is at <http://127.0.0.1:8501>. Ingest and search need no credentials or GPU, but the first run downloads the embedding model, so network is required.

### Enable research answers

Copy [`.env.example`](.env.example) to `.env` and fill in these five keys — the blanks below are reader-supplied configuration, not usable defaults:

```sh
PAPERRAG_BASE_URL=
PAPERRAG_API_KEY=
PAPERRAG_BREAKDOWN_MODEL=
PAPERRAG_RESEARCH_MODEL=
PAPERRAG_SYNTHESIS_MODEL=
```

One OpenAI-compatible Chat Completions endpoint serves all three roles, and the same model may serve each. Then run `ask`:

```sh
micromamba run -n paperrag paperrag ask "What non-parametric memory does RAG use?" --json
```

Retrieved excerpts and their metadata are sent to the configured endpoint, and provider charges may apply.

## Web interface

With the editable install above, `paperrag ui` can run from any directory: it uses the checkout's `.env` and `paperrag.toml`, and resolves relative paths (including the default `data/` and `papers/`) from the checkout root. Exported environment variables still override `.env`; files in the launch directory are not used. Non-editable installs use the launch directory.

1. **Add papers.** In **Library**, enter a PDF directory and click **Ingest directory**, then **Refresh library** to update counts.
2. **Choose a scope.** Pick papers in **Library** or search the whole collection; in **Research**, choose **All indexed papers** or **Selected papers**.
3. **Search or generate.** **Search excerpts** retrieves local evidence without credentials. **Research answer** runs the configured models: retrieved excerpts are sent to that endpoint.
4. **Inspect the evidence.** Review the answer, hypotheses and limitations, and the fact-check table. Expand a source card and use **View page [S…]** to inspect the physical PDF page or download the original; source actions reuse the saved result, not new model calls.
5. **Save your response.** With a completed **Research answer** on screen, **Save response** downloads the displayed answer, its structured analysis, fact checks, and exact indexed source excerpts as `paperrag-research-response.md`; no additional research call runs.

Editing inputs, previewing, and downloading do not initiate new research. If an original file is changed or missing, its indexed excerpts remain available but the cited page cannot be previewed or downloaded until the PDF is re-ingested.

## CLI guide

All commands use the `micromamba run -n paperrag paperrag` prefix.

| Command | Purpose |
| --- | --- |
| `download-samples` | Download the three official arXiv sample PDFs |
| `ingest PATH…` | Recursively index PDFs without changing the originals |
| `index` | Reconcile pending commits and maintain native FTS/scalar/ANN indexes |
| `stats` | Show committed corpus counts and index readiness |
| `search "QUESTION"` | Search local evidence without credentials or answer generation |
| `ask "QUESTION"` | Run deconstruction, bounded research, synthesis, and independent verification |
| `ui` | Launch the local single-user web app (default 127.0.0.1:8501) |

```sh
micromamba run -n paperrag paperrag stats --json
micromamba run -n paperrag paperrag search "What non-parametric memory does RAG use?" --top-k 5
micromamba run -n paperrag paperrag ask "How does context position affect grounded answers?" --document-id <document-id> --json
micromamba run -n paperrag paperrag ingest --force-ocr /path/to/pdfs
micromamba run -n paperrag paperrag index
```

`<document-id>` is a reader-supplied ID copied from `search --json` output or the Library. Repeated `--document-id` filters persist through research expansion. `--data-dir PATH` changes storage only, on `ingest`, `index`, `stats`, `search`, and `ask`. `stats`, `search`, and `ask` support `--json`: results go to stdout, progress and errors to stderr, failures exit nonzero, and a completed `insufficient_evidence` outcome is not an error.

## How it works

1. **Parse and chunk.** PyMuPDF native extraction with local Tesseract OCR fallback; originals are never modified. Chunks stay on one physical page, at most 384 tokenizer tokens with 64-token overlap, and canonical character offsets preserve traceability.
2. **Index and retrieve.** CPU ONNX/FastEmbed embeddings feed embedded LanceDB hybrid search (cosine plus native BM25) with reciprocal-rank reranking — no vector server. Overlap suppression and per-paper diversity cap repeated passages.
3. **Bound the evidence.** At most four follow-up queries run in one bounded expansion round, pinned to the same table version and paper scope. At most 16 whole chunks / 24,000 excerpt characters enter the evidence set, under stable source labels.
4. **Verify the answer.** Atomic claims are checked in a separate call against their original cited excerpts — not research summaries. Only supported claims that also directly answer the question, with validated nonempty excerpt quotes, become answer facts; other checks remain inspectable. Hypotheses are labeled as not established by the cited papers and paired with suggested experiments.

When nothing qualifies, the result is `insufficient_evidence`. The verification call is a model call, not an independent model or proof of scientific truth. Nonempty retrieval runs five model calls, empty retrieval runs none, and at most one corrective call repeats a check with mismatched claim-ID coverage; other provider or schema failures remain explicit errors. Research never discovers literature on the web; it stays within the indexed corpus.

## Configuration

See [`.env.example`](.env.example) and [`paperrag.example.toml`](paperrag.example.toml). Precedence: process environment &gt; `.env` &gt; TOML &gt; defaults. TOML paths are relative to the TOML file's directory; other relative paths use the working directory. Defaults: `data/` for storage, `papers/` for PDFs, and `BAAI/bge-small-en-v1.5` for embeddings.

```toml
data_dir = "./data"
pdf_dir = "./papers"
ocr_language = "eng"

[models.synthesis.options]
# Provider-specific output limit, e.g. max_completion_tokens = 4096.
```

Provider support determines the correct output-limit option. No endpoint or model is implicit; credentials belong in the environment or `.env`, never TOML, and reserved options cannot override model, messages, or credentials.

For the current cost-conscious OpenRouter test configuration, see [OPENROUTER_STAGE_MODELS.md](OPENROUTER_STAGE_MODELS.md). [PAPERRAG_SESSION_HANDOFF.md](PAPERRAG_SESSION_HANDOFF.md) records implementation state and safeguards for future agents. Neither file contains credentials.

## Data and recovery

The data directory holds `manifest.sqlite3` (revision/failure journal), `lancedb/` (authoritative chunks and native indexes), `cache/` (canonical page text and staging), and `models/` (tokenizer/CPU ONNX weights). One writer runs at a time (`write.lock`) while readers stay available; whole-document replacement is staged and reconciled on the next writer startup or `paperrag index`, closing the crash window without claiming a distributed transaction. Unchanged bytes or settings skip work; an exact byte-for-byte copy at a new path is a duplicate without alias tracking; missing originals never prune evidence. Back up the entire data directory with writers stopped, keep the original PDFs separately, and use a fresh data directory when changing embedding identity or schema.

## Limitations and troubleshooting

- **Lossy extraction.** OCR, tables, and formulas can be imperfect; inspect the original pages for evidence. Nearest neighbors and an `ok` status are not scientific or relevance guarantees.
- **OCR language data.** Tesseract resolves data from the running environment's `share/tessdata`; English (`eng`) is the default. Check with `micromamba run -n paperrag tesseract --list-langs`.
- **Empty results.** Ingest PDFs first, then check the paper scope selected in the UI or `--document-id` filters.
- **Provider errors.** Provider timeouts, schema, or truncation failures mean the endpoint cannot satisfy the request at the configured options; check endpoint/model capability and provider-specific limits. No vendor or model is recommended here.
- **Privacy.** Parsing, OCR, embeddings, search, and previews stay local; hosted research exposes retrieved excerpts and metadata to that endpoint, and initial downloads require network. The web app binds to loopback and is single-user: authentication and multi-user serving are not provided.

## Development

```sh
micromamba run -n paperrag python -m pip install -e '.[dev]'
micromamba run -n paperrag python -m pytest -q
```

Tests use generated PDFs, local Lance storage, and scripted-model fixtures; no hosted calls or downloads. [`environment.lock.yml`](environment.lock.yml) and [`requirements.lock.txt`](requirements.lock.txt) record the resolved Linux x86_64 environment; reproduce it with `micromamba create -y -f environment.lock.yml`, then `micromamba run -n paperrag python -m pip install --no-deps -e .`.