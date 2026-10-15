# PaperRAG: evidence correctness, reliability, and evaluation

## Context
Analyze the observed ASU RC and OpenRouter runs and implement the next correctness, reliability, and evaluation improvements. The selected scope retains one configured OpenAI-compatible endpoint; the plan will not add per-stage providers or a different PDF parsing stack. The intended result is accurate insufficient-evidence outcomes, inspectable primary-source support, bounded model requests, and repeatable evaluation over the existing local index and browser/CLI services.

## Observed evidence
- `data/verification/openrouter-summary.json`: the configured Claude Sonnet 4.6 / Gemini 3.1 Pro Preview / GPT-5.4 pipeline completed research in 121.00 seconds and the out-of-corpus query in 88.81 seconds. Research retained four claims and withheld ten; the clinical query incorrectly returned `ok` despite explicitly stating no clinical dose was established.
- `data/verification/live-summary.json`: ASU model authentication and a bounded JSON completion succeeded, but cross-paper research and the focused browser query timed out during deconstruction at the 180-second provider limit. The completed clinical query took 580.89 seconds and likewise returned `ok` for irrelevant NLP claims.
- `src/paperrag/llm.py` already combines system instructions into one leading message. Preserve this observed ASU compatibility fix.
- Exact inspection: nine OpenRouter research checks were locally downgraded for quote failure; the tenth insufficient check was already insufficient. Rejected quote strings for the nine were discarded, so a complete root-cause taxonomy cannot be recovered. One surviving invalid quote (C14/S2) inserts an ellipsis. Source excerpts visibly contain `ﬁ` ligatures and line-end splits such as `rele-\nvant`; these are hazards, not proven counts of the discarded failures.

- `src/paperrag/pipeline.py:25-39` appends expansion evidence to all initial sources without applying page-overlap or per-paper limits again. `retrieval.py:17-49` enforces those limits only within each retrieval/fused list; final prompt composition must enforce its own invariant while retaining stable citations.
- `pipeline.py:67-110` normalizes only whitespace for quote containment, discards invalid quote text, and renders any claim whose final verdict is `supported`. There is no independent question-relevance or answerability field in `models.py:93-147`. Exact rejected-quote causes must not be asserted if the original rejected quote text is absent from saved artifacts.
- `config.py:17-36` supports per-stage model/options dictionaries but no typed stage timeout. `llm.py:21-61` hard-codes 180 seconds and disables retries. Preserve zero automatic retries and use explicit stage options rather than inventing provider-specific output limits.

## Approach

Implement the core changes in step 1 as one result-contract cutover, including its conforming callers and fixtures; do not expose an intermediate mixture of old and new verification schemas. Steps 2 and 3 consume that completed contract. Step 4 is an independent reproducibility fix. Use the existing environment and dependencies; no storage migration or re-indexing is required.

### 1. Make factual answer eligibility explicit and bind quotes to canonical source spans

#### 1.1 Enforce one cumulative evidence budget

Reuse `RetrievalSession`, `fuse`, `diverse`, and the pinned table version in `src/paperrag/retrieval.py`. Keep the public `retrieve` and `answer` signatures unchanged.

- Extend `diverse(evidence: Sequence[Evidence], top_k: int, *, max_per_document: int = 4) -> list[Evidence]`. Factor its current same-page, more-than-half-the-shorter-span overlap predicate into `overlaps(a: Evidence, b: Evidence) -> bool` in that module and reuse it rather than implementing a second overlap convention.
- Change `pipeline.bounded_sources(candidates: Sequence[Evidence], existing: Sequence[Evidence] = (), *, max_per_document: int = 4) -> list[Evidence]` to enforce chunk-ID uniqueness, the overlap predicate, per-document counts, at most 16 chunks, and at most 24,000 total original excerpt characters across `existing + candidates`. Iterate candidates in supplied rank order; skip a whole chunk that would exceed a bound and continue looking for later fitting chunks. Never truncate an excerpt.
- At initial prompt selection pass `max_per_document=2`; retrieval itself may still return up to 12 candidates with its normal four-per-paper rule. This reserves room for expansion from the same papers and reduces the first model prompt. After the single expansion round, use the default four-per-paper bound on the cumulative set. Retain all already-issued sources and labels; append only admissible new sources. Do not replace or renumber a source already used by breakdown/research.
- Validate `existing` invariants at this internal boundary; invalid pre-existing labels/budgets are an application error, not a reason to silently drop a cited source. Sources created by the new initial selection satisfy these invariants.
- Retain validated document scope and the same pinned table throughout expansion. Do not add a similarity cutoff: nearest-neighbor score alone is not an answerability decision.
- In the saved OpenRouter result, expansion produced more than four prompt excerpts from individual papers; fix the combined selection, not just `fuse`.

#### 1.2 Add deterministic, source-derived quotation units

Create `src/paperrag/quotes.py`; the package inventory has no existing quote-selection helper. Do not change parsing, chunk text, PDF caches, embeddings, or their fingerprints.

Add these strict contracts to `src/paperrag/models.py`, reusing `Contract`:

- `QuoteSpan`: `span_id: str`, `citation: str`, `chunk_start: int`, `chunk_end: int`, `text: str`. Its offsets are Python character offsets relative to `Evidence.text`.
- `QuoteSelection`: `citation: str`, `span_ids: list[str]` with 1–8 entries. This is the **provider input contract for quote selection**, not free text returned by the model.
- Replace the old two-field result `EvidenceQuote` with `citation: str`, `quote: str`, `char_start: int`, `char_end: int`, `span_ids: list[str]`. These offsets are in the canonical physical-page text, so `char_start = evidence.char_start + chunk_start`. Result quote strings are derived locally.

Implement:

```python
build_quote_spans(evidence: Evidence) -> list[QuoteSpan]
resolve_quote_selection(
    selection: QuoteSelection,
    evidence: Evidence,
    spans: Mapping[str, QuoteSpan],
) -> list[EvidenceQuote]
```

Span construction is a lossless partition of the excerpt: start at character 0, take at most 320 characters, and, unless at the end, prefer the last paragraph boundary, then newline, then ordinary space in the final 160 characters of the window. Include the selected delimiter in the earlier span. If no boundary exists, split at the hard character boundary. Advance to the exact prior end; concatenate all span texts to recover the excerpt byte-for-character as a Python string. Empty text yields no spans. Name spans `S1:Q1`, `S1:Q2`, etc., using the stable source label and a one-based ordinal.

Resolution must:

1. Require each ID to exist in the supplied map and belong to `selection.citation`, which must be among the actual claim's citations. Reject duplicate IDs in one selection. Unknown or cross-source IDs are `StageError(code="invalid_quote_reference")` at `synthesis.verify`, not an excuse to search a different source.
2. Sort valid IDs in source order and merge adjacent selected spans into contiguous groups. Return separate quotes for nonadjacent groups; never insert ellipses or join omitted material.
3. Derive each group's exact text and page offsets from the original excerpt. Whitespace-only groups do not constitute usable support. Deduplicate repeated groups across a claim's selections.
4. Preserve ligatures, punctuation, signs, numbers, and line-end hyphens. Do not use NFKC, dehyphenation, case folding, edit distance, token similarity, or semantic matching. This removes the need for models to reproduce PDF typography without relaxing source fidelity.

The model never supplies authoritative character offsets or quote text. This avoids both off-by-one guesses and the demonstrated ellipsis/rephrasing failure.

#### 1.3 Cut over the research and verification contracts

Use these exact new fields/types in `models.py`:

- `AnswerabilityAssessment`: `coverage: Literal["answerable", "partial", "insufficient"]`, `reason: str`, `unanswered_parts: list[str]`.
- Add required `answerability: AnswerabilityAssessment` to `ResearchReport`.
- Introduce provider-output `ClaimAssessment`: `claim_id`, `verdict: Literal["supported", "contradicted", "insufficient"]`, `role: Literal["answer", "context", "insufficiency"]`, `reason`, and `evidence_spans: list[QuoteSelection]` with at most four selections. An empty list is allowed so unsupported/insufficient judgments remain expressible.
- Change `Verification` to required `answerability: AnswerabilityAssessment` and `claim_checks: list[ClaimAssessment]`.
- Retain `ClaimCheck` as the **locally validated result** type: `claim_id`, `claim_text`, `verdict`, `model_verdict` using the same verdict literal, `role` using the same role literal, `reason`, `citations: list[str]`, `evidence_quotes: list[EvidenceQuote]`, `included_in_answer: bool`, and `evidence_issue: Literal["missing_evidence"] | None`.
- Add `schema_version: Literal[2] = 2`, required `answerability: AnswerabilityAssessment`, and required `diagnostics: RunDiagnostics` to `AnswerResult`. Retain its existing two `status` values and existing question/breakdown/research/sources/models/hypotheses/limitations fields.
- Limit `DraftAnswer` to at most 12 total claims across its sections with an after-validation check. Use named constants for this, four selections per claim, eight spans per selection, 320 span characters, and the existing prompt budgets. Do not force generation settings such as temperature or a universal token-limit parameter.

The verdict remains evidence support; the role is whether the claim answers this question. Define `context` to include true but nonresponsive facts. Define `insufficiency` as a statement that the requested factual answer is unavailable, not the factual answer itself. A supported negative finding or a direct answer to an explicit question about excerpt coverage can still have role `answer`; do not classify by negative keywords or clinical terminology.

Update all in-repository construction/consumption sites, not compatibility aliases. The verified migration search is:

```text
\b(ClaimCheck|Verification|EvidenceQuote|ResearchReport|AnswerResult|StageEvent|StageError|bounded_sources|diverse|complete_json)\b
```

Run it over `src` and `tests` again before implementation. Current callers are `pipeline.py`, `llm.py`, `web.py`, `cli.py`, `ingest.py` for stage events, and the fixtures/regressions in `tests/test_pipeline.py` and `tests/test_retrieval.py`. The module-level and `LLM.complete_json(stage, messages, output_type) -> BaseModel` APIs remain unchanged. Existing free-text checker fixtures must become `ClaimAssessment`/`QuoteSelection` fixtures; remove the old runtime free-text quote path rather than accepting both wire forms.

#### 1.4 Separate question coverage, fact support, and brainstorming

In `pipeline.py`, extend the existing five-call flow; do not add a sixth model call or remove calls from nonempty-evidence runs.

- Breakdown remains an excerpt-limited paper deconstruction. Research planning still produces at most four unique nonblank local follow-ups. Research reporting now assesses answerability **after** expansion, without claiming exhaustive corpus/literature review.
- Research and draft prompts explicitly distinguish answer facts, contextual observations, missing information, and speculative experiments. When evidence is insufficient, brainstorming may propose testable ideas or evidence-collection experiments but must not present a missing factual answer from model memory as established.
- Use a single prompt-source builder that sends only `label`, `document_id`, `title`, `page_number`, and original excerpt `text` in ordinary stages. Do not send original filesystem paths, content hashes, ranking fields, or cache metadata to the provider.
- For final verification, send the original question, draft claims with their citation lists, and one deduplicated map of the **union of cited primary sources**. Represent each source by metadata plus its span IDs/text, without also sending a duplicate full-text copy. Do not repeat excerpts once per claim. Do not include breakdown or research summaries. The verifier must evaluate a claim only against its own cited source labels even though the shared map serves all claims.
- Require final answerability and per-claim roles independently of the research report. The final verifier must not label a disclaimer or unrelated supported NLP fact as the requested clinical/financial answer.
- Preserve exact-once claim-ID coverage and all existing document/citation restrictions. Reject citation markup such as `[S1]` inside draft claim text; the application alone formats citations from validated selections. Unknown references remain stage errors.

Change:

```python
validate_checks(
    draft: DraftAnswer,
    verification: Verification,
    sources: Sequence[Evidence],
    spans: Mapping[str, QuoteSpan],
) -> list[ClaimCheck]

finalize_answerability(
    research: AnswerabilityAssessment,
    verification: AnswerabilityAssessment,
    checks: Sequence[ClaimCheck],
) -> AnswerabilityAssessment
```

`validate_checks` resolves span selections, copies the original draft claim text, and preserves `model_verdict`. A model-supported or model-contradicted claim without any usable selected excerpt becomes final `insufficient` with `evidence_issue="missing_evidence"`; an invalid span reference is a stage error. Populate `citations` only from resolved quotes, in original claim citation order. Never emit an unquoted extra citation merely because the draft listed it.

Finalize deterministically:

1. Candidate factual claims require final `verdict=="supported"`, `role=="answer"`, and at least one resolved quote.
2. No candidate claims, or an `insufficient` aggregate assessment from either research or verification, yields final coverage `insufficient`. If aggregate judgments disagree, retain both in structured analysis and add warning code `answerability_disagreement`; do not promote facts on the basis of the other model.
3. Otherwise coverage is `partial` if either assessment is partial or any draft claim classified as `answer` was withheld; it is `answerable` only when both assessments are answerable and no answer-role claim was withheld. Merge `unanswered_parts` in first-seen order without duplicates.
4. Set `included_in_answer` only for candidate claims when final coverage is not insufficient. `render(draft, checks)` must consume this flag in draft section/claim order, not all supported verdicts.
5. If no claims are included, return status `insufficient_evidence` and the existing exact `INSUFFICIENT` message. Otherwise return `ok`, with partial coverage separately visible.
6. In version 2, `answer_markdown` contains only eligible factual claims and their application-formatted citations, or the exact insufficient message. Keep hypotheses and limitations in their existing separate fields; render them once in the CLI/UI. **Retain clearly labeled speculative hypotheses even for insufficient evidence**, as explicitly selected by the user. Their existence never changes factual status or coverage.

For the final assessment reason, use the fixed message `No question-responsive claim passed the evidence checks.` when no candidates remain. Otherwise, if an assessment vetoes the answer, use the first insufficient assessment's reason (research before verification), falling back to `The retrieved excerpts do not establish an answer to this question.` For a partial result caused by withheld answer-role claims, use `Some question-responsive draft claims did not pass the evidence checks.` and warning code `answer_claims_withheld`; otherwise use the first partial assessment's reason. For fully answerable output use `Question-responsive claims are supported by selected primary-source spans.` Add `answerability_disagreement` whenever the two aggregate coverage values differ.

Truly empty retrieval preserves the current zero-provider-call behavior and returns insufficient coverage, no hypotheses, and diagnostics with zero calls. Configuration, provider, malformed output, invalid reference and corrupt-index errors remain errors, not insufficient answers.

#### 1.5 Add bounded request controls and intrinsic run diagnostics

In `config.py` add `StageModel.timeout_seconds: float = Field(default=180, ge=0.1, le=900, allow_inf_nan=False)`. Support TOML `models.<stage>.timeout_seconds` and exact environment names:

```text
PAPERRAG_BREAKDOWN_TIMEOUT_SECONDS
PAPERRAG_RESEARCH_TIMEOUT_SECONDS
PAPERRAG_SYNTHESIS_TIMEOUT_SECONDS
```

Apply the existing process environment > `.env` > TOML precedence, using the same stage iteration as model IDs. Reject raw `options["timeout"]` and `options["max_retries"]` in the existing reserved-options validator so each control has one authority. Keep the other provider-specific options explicit and empty by default. Show typed timeout fields and commented provider-specific output/reasoning examples in `paperrag.example.toml`; add commented timeout keys to `.env.example`. Do not edit the user's `.env` or live TOML, change model IDs, or automatically apply OpenRouter/ASU-specific options.

In `llm.py` retain one `OpenAI` client and the single leading system message, pass `timeout=config.timeout_seconds` on each Chat Completions request, and retain `max_retries=0`. This is an SDK HTTP timeout, not a guaranteed total five-call wall-clock deadline. Keep the single enclosing JSON-fence allowance; do not add JSON repair/retry loops or mandatory native structured-output support.

Add `TokenUsage` with nullable nonnegative `prompt_tokens`, `completion_tokens`, `total_tokens`, and `LLM.last_usage: TokenUsage | None`, reset before every call and populated from standard response usage before parsing. Providers without usage report nulls; never fabricate cost/token counts. Do not serialize hidden reasoning, nested reasoning details, prompts, raw options, response objects, or API keys.

Extend `StageError` to:

```python
StageError(
    stage: str,
    message: str,
    *,
    code: str,
    step: str | None = None,
    http_status: int | None = None,
    diagnostics: RunDiagnostics | None = None,
)
```

Use the fixed codes `timeout`, `connection_error`, `provider_http_error`, `provider_error`, `empty_response`, `truncated_output`, `invalid_json`, `schema_mismatch`, `invalid_reference`, `invalid_quote_reference`, and `invalid_claim_checks`. Update every constructor located by the migration search. Include the effective timeout in timeout guidance; use safe status-specific guidance for 400/401/403/404/429 without echoing response bodies or SDK exception strings. Distinguish malformed JSON from schema mismatch. Provider refusal/filtered finish reasons are errors, not successful answers. The pipeline adds the precise step and completed diagnostics before re-raising.

Add strict result contracts:

- `CallRecord`: `step: str`, `stage: Literal["breakdown","research","synthesis"]`, configured `model: str`, output schema `contract: str`, `status: Literal["ok","error"]`, `elapsed_seconds: float` (nonnegative), effective `timeout_seconds: float`, `usage: TokenUsage | None`, `error_code: str | None`, `http_status: int | None`. All fields are required, including nullable fields; explicitly use null when unknown.
- `RunDiagnostics`: `elapsed_seconds: float` (nonnegative), `index_version: int | None`, `initial_source_count: int`, `final_source_count: int`, `excerpt_characters: int` (all counts nonnegative), `calls: list[CallRecord]`, and `warnings: list[str]`. All fields are required. `TokenUsage` fields are nullable nonnegative integers with default null.
- Extend `StageEvent` additively with `step: str | None = None`, `state: Literal["progress","started","completed","failed"] = "progress"`, and `elapsed_seconds: float | None = None`. Existing `stage`/`message` and ingestion events continue to work.

Use stable steps `local.retrieve`, `breakdown`, `research.plan`, `research.retrieve.1` through `.4`, `research.report`, `synthesis.draft`, `synthesis.verify`, `complete`. Measure durations in `Pipeline` with `time.perf_counter`; copy `LLM.last_usage` when available. Scripted test clients may legitimately have no usage metadata. Record completion only after schema/reference validation succeeds; record the failed step and preserve earlier call records on errors. Capture `RetrievalSession.version` when available; synthetic/empty sessions may report null.

Construct and close a provider client only when the pipeline owns it; an injected client remains caller-owned. Reuse that client across the five calls. Do not put credentials into Streamlit caches or automatically persist runs.

Minimal conforming browser change in this cutover: if saved answer session data lacks `schema_version==2`, clear that answer and its preview with an informational message telling the user to resubmit; never automatically rerun a paid question or upgrade an old `ok` result into a trusted answer. Search-only session results remain unaffected.

### 2. Make answer status, primary support, and runtime behavior inspectable in both surfaces

Update the existing `cli.py` and `web.py` consumers; do not add a second backend or background job framework.

- Human CLI and browser first show factual `answer_markdown`, then a partial-coverage warning when applicable, then separate `Hypotheses — not established by the cited papers` and `Limitations` sections. Hypotheses remain visible on insufficient outcomes and are explicitly outside factual verification. CLI JSON stays one direct version-2 `AnswerResult` on stdout; progress/errors stay on stderr, and expected insufficient outcomes keep exit code 0.
- Replace the current non-JSON `ask` branch's trailing full `emit(result)` with the human sections and compact support/run details, so the same answer/hypotheses are not printed again as a JSON dump. The complete serialized result remains available through `--json`.
- The fact-check table shows claim text, model/final support verdict, role, `included_in_answer`, and reason. Expand a claim to show exact locally derived quotes, citation labels and canonical physical-page offsets. Do not use raw model HTML.
- Add a `Run details` expander/table showing total/per-call elapsed time, model, timeout, token usage when supplied, final coverage, evidence counts, pinned index version and warning codes. Null usage is displayed as unavailable, not zero cost.
- Handle `StageError` separately in the browser: set the status label to the failed stable step and display its safe category/guidance plus available completed-call diagnostics. Store errors and successful results in session state without persisting credentials.
- Add `Download result JSON` for successful/insufficient completed results using the saved result only. Opening fact checks, downloading JSON, or opening a page must not initiate retrieval or another provider call.
- Keep source cards resolved by existing `source_path`/`render_page` hash checks. Do not turn quote offsets into arbitrary file paths or relax changed/missing-source notices.

### 3. Add offline-first, outcome-oriented evaluation

Create `src/paperrag/evaluation.py`, `examples/evaluation.json`, and focused tests; no evaluator or saved-run loader exists in the inspected package.

Expose the following CLI command through the existing Typer app:

```text
paperrag evaluate CASES
  (--results-dir PATH | --live)
  [--output-dir PATH]
  [--data-dir PATH]
  [--case ID]...
  [--json]
```

Exactly one of `--results-dir` and `--live` is required. `--output-dir` is required only for live mode and forbidden in offline mode. A nonempty live output directory is an error before any provider call; do not overwrite previous runs. `--case` filters by exact known IDs, deduplicated in case-file order; unknown IDs fail before inference. Missing key/index prerequisites fail before a live run. Offline mode must not load provider settings, initialize embeddings, open PDFs/indexes, or make network calls.

Use `EvaluationSuite(schema_version: Literal[1], cases: list[EvaluationCase])` in `evaluation.py`. Each case has:

```text
id                  ASCII [a-z0-9][a-z0-9_-]{0,63}, unique
question            nonblank string
expected_status     "ok" | "insufficient_evidence"
allowed_coverage    nonempty list of "answerable" | "partial" | "insufficient"
required_source_titles  list[str], default []
require_hypothesis  bool, default false
```

Implement `evaluate_results(cases: Sequence[EvaluationCase], results_dir: Path) -> EvaluationReport` and `run_evaluation(settings: Settings, cases: Sequence[EvaluationCase], output_dir: Path, on_progress: Callable[[StageEvent], None] | None = None) -> EvaluationReport`. The latter calls the same `Pipeline.answer` once per case, writes `<id>.json` or `<id>.error.json`, and writes `summary.json` before the first provider call and after every completed case. Write artifacts via temporary sibling files and atomic replacement. This surfaces unwritable output directories before inference and checkpoints completed cases. Continue independent cases after a named provider/stage failure without retrying that case; stop further paid calls on an artifact-write failure and report that I/O error. Error artifacts contain only safe code/step/status/diagnostics, never raw provider bodies. Open a fresh pinned pipeline session per case, not per expansion round.

Define `EvaluationReport` in `evaluation.py` with `schema_version: Literal[1] = 1`, `cases: list[EvaluationCaseResult]`, `passed: int`, and `failed: int` (counts, not booleans). `EvaluationCaseResult` has required `case_id: str`, `passed: bool`, `observed_status: str | None`, `observed_coverage: str | None`, `failed_checks: list[str]`, `result_path: str | None`, `error_path: str | None`, `elapsed_seconds: float | None`, and `usage: TokenUsage | None`. Nullable fields are explicit nulls. Sum a token-usage field only when every recorded call supplies that field; otherwise that aggregate field stays null. Do not guess monetary cost or add an automatic LLM judge.

Offline checks must cover:

1. Result schema version 2, matching question, expected status/allowed coverage, and coherent `included_in_answer` flags.
2. Every included claim is supported, role `answer`, and backed by a quote whose original-text slice, page offsets and selected span IDs resolve exactly against its saved source. Reuse the span resolver. This verifies saved-source consistency, not independent scientific truth or the availability of the original PDF.
3. Required paper titles occur in sources actually cited by **included** claims, not merely anywhere in retrieval results. All sources/labels and claim/check IDs are unique and references valid.
4. The global source budgets, paper cap and overlap invariant hold; source metadata cannot invent a different quotation.
5. Hypotheses have nonblank experiments when required and remain separate from the exact insufficient factual message. Their presence must not promote status to `ok`.
6. A legacy result without version 2, missing file, corrupt JSON, or invalid result contract is an explicit failed case (`incompatible_result`, `missing_result`, or `invalid_result`), not silent migration or an inference trigger. If both `<id>.json` and `<id>.error.json` exist, report `invalid_result` rather than choosing one. Existing historical files remain untouched and are not trusted under the new contract. Ignore artifacts for unselected case IDs.

Use exit code 0 only when all selected cases pass; 1 for completed evaluation failures including case/provider failures; 2 for invalid arguments/case input/prerequisites. With `--json`, stdout contains only the report; live stage progress remains stderr.

Populate the case file with these six exact cases and questions, using the titles already present in saved sources:

| ID | Question | Expected outcome |
| --- | --- | --- |
| `rag-memory` | What non-parametric memory does the RAG model use? | `ok`, coverage answerable or partial, included citation to `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks` |
| `context-position` | How does the position of relevant information affect question-answering performance in Lost in the Middle? | `ok`, answerable or partial, included citation to `Lost in the Middle: How Language Models Use Long Contexts` |
| `research` | How do retrieval-augmented generation and context position affect evidence-grounded question answering, and what experiment would distinguish their limitations? | `ok`, answerable or partial, included citations to both titles above, at least one separately labeled hypothesis/experiment |
| `abstention` | What dose of amoxicillin should be prescribed to treat community-acquired pneumonia in a child? | `insufficient_evidence`, coverage insufficient, at least one separately labeled hypothesis/experiment; no verified factual dose |
| `unrelated-finance` | What was Acme Robotics' audited net profit in fiscal year 2025? | `insufficient_evidence`, coverage insufficient |
| `mixed` | What non-parametric memory does RAG use, and what dose of amoxicillin should be prescribed for pneumonia in a child? | `ok`, coverage partial, included RAG citation, an explicit unanswered clinical part, and at least one separately labeled hypothesis/experiment |

Automated checks establish contract/provenance outcomes, not paraphrase wording or scientific correctness. Do not score by required answer substrings, term counts, or number of generated claims. Inspect the actual primary passages and relevance manually in live verification.

### 4. Repair the observed nonportable dependency pin

`requirements.lock.txt:41` currently uses `packaging @ file:///home/conda/feedstock_root/.../work`; this cannot reproduce the environment on another workstation. Replace that one entry with `packaging==26.3`, matching the verified `environment.lock.yml:46` conda version. Keep `-e .`, all other resolved pins and the existing environment; do not resolve a different dependency set or install globally. This is a reproducibility correction for running evaluation, not a framework/dependency upgrade.

## Critical files and anchors

- `src/paperrag/pipeline.py`, `bounded_sources`, `validate_checks`, `render`, `Pipeline.answer`: the current support-only success decision and cross-round selection are the correctness boundaries.
- New `src/paperrag/quotes.py`, `resolve_quote_selection`: displayed quotes must be exact original slices, never reconstructed model text or normalized replacements.
- `src/paperrag/llm.py`, `LLM.complete_json` and `StageError`: preserve single-system-message compatibility, explicit model selection, zero retries and safe error metadata.
- New `src/paperrag/evaluation.py`, live/offline entry points: offline evaluation must never become a provider call or silently accept legacy trust decisions.

## Verification

### Preconditions and baseline preservation

Work from `/home/kstv/PaperRAG`. All Python/application/test commands use `micromamba run -n paperrag`; use the existing environment and retain the three downloaded papers and committed index. Initial observed baseline is three documents, 52 physical pages and 157 chunks. No code or model calls were executed during planning. Do not read/print credentials; use `load_settings()` and report model IDs/presence only. If live keys/models are unavailable or endpoint access fails, finish deterministic/offline checks and report the exact unexercised live prerequisite without substituting a provider.

### Deterministic consumer regressions

Extend the existing `tests/test_pipeline.py` and `tests/test_retrieval.py`, reusing `ScriptedLLM`, `SessionSpy`, generated PDFs and tiny vectors. Create `tests/test_quotes.py`, `tests/test_config.py`, `tests/test_llm.py`, `tests/test_evaluation.py`, and `tests/test_cli.py` for new behavior; no hosted credentials, real downloads or ONNX model loads in tests.

- Reproduce both saved failure patterns with tiny fixtures: supported unrelated context and supported insufficiency disclaimers must not produce `ok`; five scripted calls still run and labeled hypotheses survive an insufficient result. Include a legitimate negative scientific finding and an explicit meta-coverage question that can be answered, to reject domain/negative-keyword heuristics.
- Positive answer, partial mixed question, aggregate-assessment disagreement, no eligible claims, empty corpus/no calls, and provider failure distinct from insufficiency. A hypothesis or limitation must never satisfy factual status.
- Exact span partitions for ligatures, line-end hyphens, Unicode, paragraph boundaries, long unbroken text, blank/empty excerpts and final short fragments. Resolved text/offsets must equal original slices. Unknown/cross-source/duplicate IDs fail; absent/whitespace-only selections cannot support a claim. Contradicted/insufficient claims are never included. Unknown claim/source references and duplicate/missing check IDs retain named errors.
- An initially omitted but follow-up-retrieved answer can become eligible. Initial at-most-two/per-paper and cumulative at-most-four/per-paper bounds, overlap across rounds, whole-chunk character limits, stable labels and document filters must all hold.
- Missing provider usage is null; usage from a schema-invalid response may still be recorded safely. Timeout overrides obey process > `.env` > TOML, invalid nonfinite/out-of-range timeouts and raw override keys are rejected, timeout/auth/rate-limit/schema errors are distinct, failures do not retry or switch models, and failure diagnostics retain completed calls without exposing key/raw body/hidden reasoning.
- Offline evaluation must work without LLM settings, never invoke a provider, measure only included-claim citation coverage, fail on a true-but-irrelevant `ok` result, and distinguish missing/corrupt/legacy artifacts. Live-mode tests use deterministic pipeline fixtures only and verify outcomes/errors/artifacts, not copied request wiring.
- CLI JSON output is parseable and unmixed with progress; insufficient answers exit 0, evaluation failures exit 1, and invalid evaluation modes exit 2.
- Replace the obsolete free-text quote regression with missing/invalid selection regressions; do not retain a legacy quote acceptance path. Delete existing wording-only assertions and the isolated mock model-forwarding assertion at `tests/test_pipeline.py:353` rather than pinning implementation details.

Run the full permanent suite once after integration:

```sh
micromamba run -n paperrag python -m pytest -q
micromamba run -n paperrag python -m pip check
micromamba run -n paperrag python -m pip install --dry-run --no-deps -r requirements.lock.txt
```

The lock dry-run must not refer to the absent conda feedstock filesystem. These checks do not substitute for the following runtime proof.

### Runtime request/selection smoke proof

Use a throwaway local HTTP server through the installed OpenAI client and actual `LLM.complete_json` to exercise a configured 0.1-second timeout against a delayed response, then a bounded successful JSON response with usage. Observe the named timeout category, no automatic retry and a valid recorded completion with no reasoning/credential fields. Also run one controlled invalid-span response through the real pipeline boundary; it must end as `invalid_quote_reference`, never an answer. Do not connect test fixtures to normal runtime/demo paths.

Run `paperrag search` in a fresh environment process for the existing RAG-memory question and confirm committed evidence remains available. Run `paperrag stats --json` before/after all query verification: documents/pages/chunks and original source files must remain unchanged.

### Live evaluation and offline replay

When the real three-sample index and explicitly configured credentials are available:

```sh
micromamba run -n paperrag paperrag evaluate examples/evaluation.json \
  --live --output-dir data/verification/hardening-run --json
micromamba run -n paperrag paperrag evaluate examples/evaluation.json \
  --results-dir data/verification/hardening-run --json
```

Use a new empty output directory if that name already exists; never overwrite historical ASU/OpenRouter artifacts. The offline replay must reproduce the same case pass/fail decisions with provider access disabled. All nonempty-evidence live cases retain five call records, including out-of-corpus brainstorming. Required outcomes are the table above; do not weaken negative or partial expectations to match model mistakes.

Manually inspect the RAG Wikipedia-index passage, Lost in the Middle's beginning/end-versus-middle finding, and the joint experiment hypothesis. Confirm included claims answer the question and every displayed quote exactly equals its canonical excerpt slice. Report observed coverage, call durations, usage availability and failures; do not claim improvement merely because old rejected quotes were discarded or because source titles occurred somewhere in retrieval. If a live case fails, preserve diagnostics and fix the general prompt/validation defect, not a question-specific exception or looser evaluator.

### Browser proof

Launch `micromamba run -n paperrag paperrag ui --port 8502` on loopback for verification so an existing service need not be interrupted. Use the real browser:

1. Observe the existing library and run the RAG-memory research question once with configured models.
2. Inspect factual coverage, claim roles/inclusion, exact span-derived quotes and page offsets, partial warnings when applicable, and the `Run details` table.
3. Open a cited physical page and download result JSON; confirm the same saved diagnostics/call count remains and no paid query repeats.
4. Run the out-of-corpus question and observe the exact insufficient factual message plus separately labeled speculative hypotheses. It must not show unrelated NLP facts as the answer or use `ok` solely because brainstorming exists.
5. Exercise a controlled provider-error/local-test endpoint without changing the user's real `.env`; observe the failed step and safe diagnostics, not a successful insufficient answer. Clear a version-1 saved session fixture and confirm an informational resubmit message without automatic inference.

Capture actual UI proof and close the test tab. No UI-test framework or new frontend dependency is required.

## Assumptions and contingencies
The user selected correctness, reliability, and evaluation as the extension scope. Retain the single-endpoint architecture, current local embeddings, and existing source PDFs/index; do not silently change model IDs, introduce automatic provider fallback, or re-index scientific documents as part of these fixes.

The user selected continued, explicitly labeled brainstorming for nonempty but insufficient evidence: keep the full five-call path and retain speculative hypotheses separately from factual answer status. Do not introduce an early-abstention shortcut or count a hypothesis, disclaimer, or contextual fact as a supported answer.
