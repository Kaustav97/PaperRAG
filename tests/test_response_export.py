"""Deterministic Markdown export coverage for completed research responses."""
from paperrag.models import (
    AnswerResult,
    Breakdown,
    ClaimCheck,
    CitedText,
    Evidence,
    EvidenceQuote,
    Hypothesis,
    PaperBreakdown,
    ResearchReport,
)
from paperrag.pipeline import INSUFFICIENT
from paperrag.response_export import render_response_markdown


DOC = "d" * 64
EXCERPT = "First line — naïve χ² test.\n\nSecond paragraph: é, ü, 中, अ.\nLast line."


def source(label: str, *, page: int, text: str = EXCERPT) -> Evidence:
    return Evidence(
        source_path="/papers/history.pdf",
        title="A history of memory",
        page_number=page,
        chunk_ordinal=0,
        char_start=40,
        char_end=120,
        extraction_method="native",
        text=text,
        chunk_id="c" * 64,
        document_id=DOC,
        content_sha256="f" * 64,
        rank=1,
        score=0.75,
        label=label,
    )


def ok_result() -> AnswerResult:
    return AnswerResult(
        status="ok",
        question="What non-parametric memory does the RAG model use?",
        answer_markdown="The dense vectors **answer** the question [S1].",
        breakdown=Breakdown(
            subquestions=["What was observed?", "How was it measured?"],
            papers=[PaperBreakdown(
                document_id=DOC,
                problem="A problem",
                method="A method",
                findings="A finding",
                limitations="A limit",
                citations=["S1"],
            )],
            gaps=["scope outside the excerpts"],
        ),
        research=ResearchReport(
            findings=[CitedText(text="The reported finding.", citations=["S1"])],
            agreements_conflicts=[CitedText(text="A conflicting view.", citations=["S2"])],
            remaining_gaps=["exact dose not established"],
            hypotheses=[Hypothesis(text="A working hypothesis.", suggested_experiment="Measure it.", citations=["S1"])],
        ),
        claim_checks=[ClaimCheck(
            claim_id="c1",
            verdict="supported",
            role="answer",
            reason="The excerpt states it.",
            claim_text="The answer claim.",
            evidence_quotes=[EvidenceQuote(citation="S1", quote="First line — naïve χ² test.")],
        )],
        sources=[source("S1", page=2), source("", page=3, text="Second excerpt.\n\nAlso preserved.")],
    )


def test_ok_answer_renders_ordered_sections():
    rendered = render_response_markdown(ok_result())
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")
    headings = [
        "# PaperRAG research response",
        "## Question",
        "## Status",
        "## Answer",
        "## Structured deconstruction and research",
        "### Breakdown",
        "### Research",
        "## Fact checks",
        "## Sources",
        "### [S1] A history of memory",
    ]
    positions = [rendered.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_ok_answer_contains_answer_and_structured_content():
    rendered = render_response_markdown(ok_result())
    assert "What non-parametric memory does the RAG model use?" in rendered
    assert "The dense vectors **answer** the question [S1]." in rendered
    assert "`ok`" in rendered
    assert "- What was observed?" in rendered
    assert "- How was it measured?" in rendered
    assert f"#### {DOC}" in rendered
    assert "**Problem:** A problem" in rendered
    assert "**Method:** A method" in rendered
    assert "**Findings:** A finding" in rendered
    assert "**Limitations:** A limit" in rendered
    assert "**Citations:** [S1]" in rendered
    assert "- scope outside the excerpts" in rendered
    assert "- The reported finding." in rendered
    assert "  Citations: [S1]" in rendered
    assert "- A conflicting view." in rendered
    assert "  Citations: [S2]" in rendered
    assert "- exact dose not established" in rendered
    assert "- A working hypothesis." in rendered
    assert "  Suggested experiment: Measure it." in rendered


def test_ok_answer_renders_literal_claim_check():
    rendered = render_response_markdown(ok_result())
    assert "### c1" in rendered
    assert "**Claim:** The answer claim." in rendered
    assert "**Verdict:** `supported`" in rendered
    assert "**Role:** `answer`" in rendered
    assert "**Reason:** The excerpt states it." in rendered
    assert "**Evidence:**" in rendered
    assert "- First line — naïve χ² test. [S1]" in rendered


def test_sources_render_full_metadata_and_exact_excerpt():
    rendered = render_response_markdown(ok_result())
    assert "### [S1] A history of memory" in rendered
    assert "### 2 A history of memory" in rendered
    assert "**Source path:** /papers/history.pdf" in rendered
    assert "**Physical page:** 2" in rendered
    assert "**Extraction method:** native" in rendered
    assert "**Character range:** 40–120" in rendered
    assert f"**Chunk ID:** {'c' * 64}" in rendered
    assert f"**Document ID:** {DOC}" in rendered
    assert f"**Content SHA-256:** {'f' * 64}" in rendered
    assert "**Rank:** 1" in rendered
    assert "**Score:** 0.75" in rendered
    assert "**Excerpt:**" in rendered
    assert "    First line — naïve χ² test." in rendered
    assert "    Second paragraph: é, ü, 中, अ." in rendered
    expected_block = "\n".join("    " + line if line else "" for line in EXCERPT.split("\n"))
    assert rendered.count(expected_block) == 1


def test_insufficient_result_emits_hypotheses_and_limitations_once_outside_answer():
    result = AnswerResult(
        status="insufficient_evidence",
        question="What dose should be prescribed?",
        answer_markdown=INSUFFICIENT,
        hypotheses=[Hypothesis(text="Maybe a higher dose.", suggested_experiment="Run a trial.", citations=["S1"])],
        limitations=["No dose studied in the excerpts."],
    )
    rendered = render_response_markdown(result)
    assert rendered.count(INSUFFICIENT) == 1
    assert rendered.count("## Hypotheses — not established by the cited papers") == 1
    assert rendered.count("## Limitations") == 1
    assert "Maybe a higher dose." in rendered
    assert "  Suggested experiment: Run a trial." in rendered
    assert "  Citations: [S1]" in rendered
    assert "- No dose studied in the excerpts." in rendered
    assert "## Fact checks\n\nNone." in rendered
    assert "## Sources\n\nNone." in rendered


def test_ok_answer_never_emits_insufficient_only_sections():
    rendered = render_response_markdown(ok_result())
    assert "## Hypotheses — not established by the cited papers" not in rendered
    assert "## Limitations" not in rendered