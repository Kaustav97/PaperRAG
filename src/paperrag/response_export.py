"""Deterministic Markdown export of a completed research response.

Pure formatting boundary: no source lookup, PDF opening, network call,
timestamp, or secret. The rendered text is derived only from the supplied
``AnswerResult`` and ends with exactly one newline.
"""
from __future__ import annotations

from paperrag.models import AnswerResult, CitedText, Evidence, Hypothesis, PaperBreakdown


def _citations_label(citations: list[str]) -> str:
    if not citations:
        return ""
    return "[" + ", ".join(citations) + "]"


def _cited_text_lines(item: CitedText) -> list[str]:
    lines = [f"- {item.text}"]
    label = _citations_label(item.citations)
    if label:
        lines.append(f"  Citations: {label}")
    return lines


def _hypothesis_lines(item: Hypothesis) -> list[str]:
    lines = [f"- {item.text}", f"  Suggested experiment: {item.suggested_experiment}"]
    label = _citations_label(item.citations)
    if label:
        lines.append(f"  Citations: {label}")
    return lines


def _render_breakdown(result: AnswerResult) -> list[str]:
    parts = ["### Breakdown"]
    breakdown = result.breakdown
    if breakdown is None:
        return parts
    if breakdown.subquestions:
        parts.append("**Subquestions:**")
        parts.extend(f"- {subquestion}" for subquestion in breakdown.subquestions)
    for paper in breakdown.papers:
        parts.append(f"#### {paper.document_id}")
        parts.append(f"**Problem:** {paper.problem}")
        parts.append(f"**Method:** {paper.method}")
        parts.append(f"**Findings:** {paper.findings}")
        parts.append(f"**Limitations:** {paper.limitations}")
        label = _citations_label(paper.citations)
        if label:
            parts.append(f"**Citations:** {label}")
    if breakdown.gaps:
        parts.append("**Gaps:**")
        parts.extend(f"- {gap}" for gap in breakdown.gaps)
    return parts


def _render_research(result: AnswerResult) -> list[str]:
    parts = ["### Research"]
    research = result.research
    if research is None:
        return parts
    if research.findings:
        parts.append("**Findings:**")
        for item in research.findings:
            parts.extend(_cited_text_lines(item))
    if research.agreements_conflicts:
        parts.append("**Agreements/conflicts:**")
        for item in research.agreements_conflicts:
            parts.extend(_cited_text_lines(item))
    if research.remaining_gaps:
        parts.append("**Remaining gaps:**")
        parts.extend(f"- {gap}" for gap in research.remaining_gaps)
    if research.hypotheses:
        parts.append("**Hypotheses:**")
        for item in research.hypotheses:
            parts.extend(_hypothesis_lines(item))
    return parts


def _render_fact_checks(result: AnswerResult) -> list[str]:
    parts = ["## Fact checks"]
    if not result.claim_checks:
        parts.append("None.")
        return parts
    for check in result.claim_checks:
        parts.append(f"### {check.claim_id}")
        parts.append(f"**Claim:** {check.claim_text}")
        parts.append(f"**Verdict:** `{check.verdict}`")
        parts.append(f"**Role:** `{check.role}`")
        parts.append(f"**Reason:** {check.reason}")
        if check.evidence_quotes:
            parts.append("**Evidence:**")
            parts.extend(f"- {quote.quote} [{quote.citation}]" for quote in check.evidence_quotes)
        else:
            parts.append("**Evidence:** none")
    return parts


def _indent_block(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line else "" for line in text.split("\n"))


def _render_sources(result: AnswerResult) -> list[str]:
    parts = ["## Sources"]
    if not result.sources:
        parts.append("None.")
        return parts
    for number, evidence in enumerate(result.sources, 1):
        heading_label = f"[{evidence.label}]" if evidence.label else str(number)
        parts.append(f"### {heading_label} {evidence.title}")
        parts.append(f"**Source path:** {evidence.source_path}")
        parts.append(f"**Physical page:** {evidence.page_number}")
        parts.append(f"**Extraction method:** {evidence.extraction_method}")
        parts.append(f"**Character range:** {evidence.char_start}–{evidence.char_end}")
        parts.append(f"**Chunk ID:** {evidence.chunk_id}")
        parts.append(f"**Document ID:** {evidence.document_id}")
        parts.append(f"**Content SHA-256:** {evidence.content_sha256}")
        parts.append(f"**Rank:** {evidence.rank}")
        parts.append(f"**Score:** {evidence.score}")
        parts.append("**Excerpt:**")
        parts.append(_indent_block(evidence.text))
    return parts


def render_response_markdown(result: AnswerResult) -> str:
    parts = [
        "# PaperRAG research response",
        "## Question",
        result.question,
        "## Status",
        f"`{result.status}`",
        "## Answer",
        result.answer_markdown,
    ]
    if result.status == "insufficient_evidence":
        if result.hypotheses:
            parts.append("## Hypotheses — not established by the cited papers")
            for hypothesis in result.hypotheses:
                parts.extend(_hypothesis_lines(hypothesis))
        if result.limitations:
            parts.append("## Limitations")
            parts.extend(f"- {limitation}" for limitation in result.limitations)
    parts.append("## Structured deconstruction and research")
    parts.extend(_render_breakdown(result))
    parts.extend(_render_research(result))
    parts.extend(_render_fact_checks(result))
    parts.extend(_render_sources(result))
    return "\n\n".join(parts) + "\n"