# PaperRAG

Local PDF parsing/OCR, persistent hybrid retrieval, and evidence-checked research through one explicitly configured OpenAI-compatible endpoint. Python 3.11; CPU-only ONNX embeddings; no vector server or JavaScript build.

## Install

```sh
micromamba create -y -f environment.yml
micromamba run -n paperrag python -m pip install -e '.[dev]'
micromamba run -n paperrag python -m pip check
micromamba run -n paperrag tesseract --list-langs
```

The application resolves OCR data from the running environment's `share/tessdata`, not the system installation. English `eng.traineddata` is required for English scanned pages.

`environment.lock.yml` records the resolved Linux x86_64 conda builds and Python packages; `requirements.lock.txt` records the working pip pins. To reproduce that resolution, create the environment from `environment.lock.yml`, then install this checkout with `micromamba run -n paperrag python -m pip install --no-deps -e .`. The lock deliberately excludes a registry release of PaperRAG; this source checkout is the application.

## Configuration

Copy `paperrag.example.toml` to `paperrag.toml` and `.env.example` to `.env` as needed. No model or endpoint is selected implicitly. Process environment overrides `.env`; both override TOML. TOML paths are relative to the TOML file; other relative paths are relative to the working directory. `--data-dir` changes storage only.

All LLM settings are optional for local ingest/search. Research requires `PAPERRAG_BASE_URL`, `PAPERRAG_API_KEY`, and explicit `PAPERRAG_BREAKDOWN_MODEL`, `PAPERRAG_RESEARCH_MODEL`, `PAPERRAG_SYNTHESIS_MODEL`. The same model may fill all roles. Keep credentials in the environment or `.env`, never TOML. Do not send secrets in chat.

**Privacy:** retrieved excerpts and metadata sent to a hosted inference endpoint leave this machine. Parsing, OCR, embeddings, retrieval, and source previews run locally. Initial sample/model downloads require network access. Research is bounded to the indexed corpus; it does not discover literature on the web. Bind the single-user browser app to loopback; authentication and multi-user serving are not provided.

Provider-specific Chat Completions arguments belong under each `[models.<stage>.options]`. Defaults are empty; the app does not force temperature, JSON-schema support, or a vendor-specific token limit. Reserved model/messages/stream/credential arguments are rejected.

## Use

```sh
micromamba run -n paperrag paperrag download-samples
micromamba run -n paperrag paperrag ingest papers
micromamba run -n paperrag paperrag stats --json
micromamba run -n paperrag paperrag search 'What non-parametric memory does RAG use?' --json
micromamba run -n paperrag paperrag ask 'How does context position affect grounded answers?' --json
micromamba run -n paperrag paperrag ui
```

Open <http://127.0.0.1:8501>. Library recursively indexes local directories, paginates documents and preserves selections across pages. Research supports all or selected papers, credential-free excerpt search, and explicitly submitted research calls. Opening a source uses saved session results, not another paid query. Use Refresh library after CLI ingestion. Source cards show exact indexed excerpts and physical pages; previews/downloads require the original file to match its indexed SHA-256.

`ingest`, `index`, `stats`, `search`, and `ask` accept `--data-dir PATH`. `search` accepts `--top-k 1..50`; `search` and `ask` accept repeated `--document-id HEX` filters, including during expansion. `stats`, `search`, and `ask` support `--json` (stdout only; progress/errors are stderr). Errors exit nonzero; a completed insufficient-evidence outcome is not an error. `ingest --force-ocr PATH` overrides scan detection. `paperrag index` reconciles pending commits and maintains native indexes independently.

## Evidence and recovery

- Native extraction preserves text blocks and paragraph breaks but cannot guarantee faithful scientific table/formula reconstruction. Inspect original pages for evidence. Scanned pages use local Tesseract at 200 DPI; truly blank pages are counted, not embedded.
- Every chunk stays on one physical PDF page, at most 384 tokenizer tokens with 64-token overlap. Character offsets slice the canonical normalized page text stored in `data/cache/`. Search is hybrid cosine + native BM25, reciprocal-rank reranked with overlap suppression and at most four chunks per paper. Nearest neighbors alone do not establish relevance.
- `manifest.sqlite3` (WAL) journals successful revisions and failures; `lancedb/` holds authoritative chunks; `cache/` holds canonical pages and complete Arrow staging; `models/` holds downloaded tokenizer/CPU ONNX weights. Originals are never moved or changed. Back up the entire data directory with writers stopped.
- Only one writer runs (`write.lock`); readers remain available. Parse/embed failures leave prior evidence intact. Whole-paper atomic merges delete only obsolete chunks belonging to that document. Installed LanceDB 0.40 uses the unqualified `document_id` column in the source-missing delete predicate; `target.document_id` is not accepted by that API. SQLite pending entries reconcile against committed Lance hashes/fingerprints on the next writer startup, closing the cross-database crash window without claiming a distributed transaction.
- Unchanged bytes/settings skip work. A new path containing exact indexed bytes is reported as a duplicate without alias tracking. Changed indexed paths always replace their own revision. Missing originals do not prune evidence. Choose a new data directory to change embedding identity/schema; silent vector-space mixing is forbidden.
- Native FTS/scalar indexes are maintained on ingestion and every 20 commits. Below 10,000 chunks, vector search is exact; at that threshold `IVF_PQ` uses cosine, 24 subvectors and bounded partitions. Maintenance errors are explicit; newly added rows are never excluded via fast-search shortcuts.

## Research contract

The local index is stage 1. Stage 2 deconstructs excerpts; stage 3 plans at most four follow-up queries and performs one bounded expansion round against the same pinned table version and paper filter, then reports research; stage 4 drafts atomic claims and separately checks each against primary excerpts. Up to 16 whole chunks / 24,000 excerpt characters enter the evidence set. Source labels remain stable across the run.

Only `supported` claims with valid nonempty verbatim excerpt quotes and role `answer` are emitted as answer facts. The verifier separately assigns question-relative roles: `answer` directly responds to the question, `context` is background/nonresponsive information, and `insufficiency` says the requested answer is unavailable. Supported negative findings and direct answers to explicit excerpt-coverage questions can still be `answer`. Context, disclaimers, and contradicted/insufficient claims remain visible in fact checks, not as answer facts. If nothing qualifies, the result is `insufficient_evidence` with the exact insufficient-evidence message; hypotheses and limitations remain available separately. Nonempty evidence still runs all five model calls; empty retrieval runs none. Hypotheses are explicitly labeled as not established by the cited papers and paired with experiments. The checker evaluates support and relevance against retrieved excerpts, **not independent scientific truth**; model-based relevance judgments remain fallible. Untrusted paper text cannot execute tools or override reference validation. Missing/unknown roles, invalid references, malformed/truncated JSON and provider failures are explicit stage errors, never successful fabricated answers or automatic retries.

For gateways requiring a single system message, the app combines its JSON schema and safety instructions into one leading system message. This avoids ASU RC's `System message must be at the beginning` rejection without changing the selected endpoint or model.

## Development

```sh
micromamba run -n paperrag python -m pytest -q
```

Permanent tests use temporary generated PDFs, actual native Lance storage and deterministic tiny-vector/scripted-model fixtures. They do not download papers/models or call a hosted provider. Runtime never uses those fixtures.

## Exercised verification

The dedicated environment passed `pip check`; its console shebang points to its own Python and its Tesseract includes English data. Official samples were downloaded with recorded hashes/provenance, then indexed in 33.591 seconds including initial model downloads.

| Indexed sample | Physical pages | Chunks |
| --- | ---: | ---: |
| Attention Is All You Need | 15 | 36 |
| Retrieval-Augmented Generation | 19 | 61 |
| Lost in the Middle | 18 | 60 |
| Total | 52 | 157 |

Fresh-process queries retrieved RAG's Wikipedia dense-vector-index/neural-retriever description on physical page 2 and Lost in the Middle's beginning/end-versus-middle finding on physical page 1. Every returned excerpt was checked against committed chunk IDs, canonical character offsets and physical page bounds. Unchanged ingest and a new-path duplicate preserved the complete chunk-ID set. Writer contention failed explicitly while readers remained usable.

The offline suite passed 30 tests; the six retrieval tests passed again after their FTS-isolation and page-provenance fixtures were refined. A real image-only copy of Attention's first page exercised automatic scan detection, local OCR and retrieval with page-1 `ocr` provenance; the title and readable abstract text were compared visually with the original image.

The synthetic ANN smoke used 10,048 rows derived from the real sample vectors, with seeded 1e-4 Gaussian perturbations to disambiguate repeated-vector ties. These are **not additional research papers**. Actual `paperrag index` built ready native FTS/scalar/IVF_PQ indexes (39 partitions), taking 2.28 seconds and 335,152 KiB peak child RSS measured through `getrusage` (`/usr/bin/time` was unavailable). Exact-ID mean recall@10 across ten fixed sample-derived questions was 0.41 with 20 probes/refinement 5, unchanged with all partitions/refinement 5, and 0.80 with all partitions/refinement 20. Individual tuned recalls were eight 1.0 and two 0.0; this deliberately clustered synthetic check does not establish real-corpus relevance or large-scale performance. Production retrieval now probes every partition (the build ceiling of 256 is clamped to the index's actual count) with refinement 20, trading speed for recall. The retained query parameters and actual hybrid path were exercised again.

A real browser verified library counts after CLI ingestion without restarting the service, selected-paper retrieval, the exact RAG excerpt and rendered physical page 2, and explicit missing-configuration errors. Source actions use session-owned results. The browser tabs were closed.

### Live ASU RC verification

Using the configured ASU RC endpoint and `qwen35-397b-a17b` for all three stages, authenticated model listing and a bounded JSON completion succeeded. The initial full request exposed a multiple-system-message incompatibility, now fixed by combining application instructions into one leading system message. The offline suite still passed all 30 tests.

With the original model options unchanged, the cross-paper research query timed out at the deconstruction stage's 180-second provider timeout (184.54 seconds total CLI time). A clinical dosing query completed all five model calls in 580.89 seconds and produced no treatment dose. However, it failed the expected abstention outcome: unrelated but excerpt-supported NLP claims survived verification, producing status `ok` instead of `insufficient_evidence`. The saved result is `data/verification/live-abstention.json` (runtime artifact, not committed).

After restarting the browser service with the compatibility fix, a focused, RAG-paper-only question also reached deconstruction but timed out at the provider's 180-second limit. The browser displayed the named stage error. Sanitized outcomes are recorded in `data/verification/live-summary.json`; `.env` and its model options were not changed.

The failed clinical acceptance check motivated the question-relevance gate described above. Quote support alone no longer makes a claim eligible for the factual answer: the verifier must also classify it as directly responsive. This is a model-based safeguard, not proof of universal answerability or relevance.

### Question-relevance gate verification

The updated offline suite passed all 40 tests, including supported-but-nonresponsive claims, mixed answer/context claims, responsive negative findings, excerpt-coverage questions, invalid quotes, strict role-schema failures, and Streamlit saved-result compatibility. A real browser also displayed the stale-answer clearing notice without resubmitting research.

Live CLI checks used the unchanged configured models: `anthropic/claude-sonnet-4.6` for breakdown, `google/gemini-3.1-pro-preview` for research, and `openai/gpt-5.4` for synthesis.

| Question | Status | CLI elapsed |
| --- | --- | ---: |
| What non-parametric memory does the RAG model use? | `ok` | 83.69 s |
| What dose of amoxicillin should be prescribed to treat community-acquired pneumonia in a child? | `insufficient_evidence` | 82.36 s |

Both runs completed all five model calls. The factual result emitted the Wikipedia dense-vector-index answer with a verbatim quote from the RAG paper's physical page 2; five supported background claims remained only in fact checks. The medical result emitted the exact insufficient-evidence message, excluded a supported context claim, and retained two separately stored hypotheses plus limitations. Full runtime results are saved in `data/verification/role-gate-factual.json` and `data/verification/role-gate-abstention.json`. These two successful scenarios do not establish universal relevance accuracy.

## Using the web interface

Start the app from the project directory:

```sh
micromamba run -n paperrag paperrag ui
```

Open <http://127.0.0.1:8501>, then:

1. **Add papers:** In **Library**, enter a local PDF directory and click **Ingest directory**. Click **Refresh library** afterward to update counts and the paper list.
2. **Choose a scope:** Select paper checkboxes in **Library**, or search the whole collection. In **Research**, enter your question and choose **All indexed papers** or **Selected papers**.
3. **Search or generate:** **Search excerpts** retrieves local evidence without API credentials. **Research answer** runs the configured models and requires the endpoint, key, and three model IDs in `.env`; retrieved excerpts are sent to that endpoint.
4. **Inspect the evidence:** Review the answer, separately labeled hypotheses, limitations, and fact-check table. Expand a source card and click **View page [S…]** to inspect the physical PDF page or download the original. Source actions reuse the saved result rather than repeating model calls.
5. **Save your response:** With a completed **Research answer** on screen, click **Save response** to download the currently displayed answer, its structured analysis, fact checks, and exact indexed source excerpts as `paperrag-research-response.md`. The download is generated from the saved answer and performs no additional research call.

If a source is missing or changed, its indexed excerpt remains available but the original page cannot be shown as the cited version; re-ingest the PDF to refresh it. Saved browser answers from before question-relevance checks are cleared with a notice, including their preview selection; submit the question again explicitly to generate a replacement. Search-only results and answers with no checks are unaffected. A result marked `ok` is not proof of relevance—check that the cited passages actually answer your question.
