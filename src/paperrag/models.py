"""Strict data contracts shared by storage, research, CLI and UI."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PageText(Contract):
    source_path: str
    title: str
    page_number: int = Field(ge=1)
    text: str
    extraction_method: Literal["native", "ocr"]
    metadata: dict[str, str | None] = Field(default_factory=dict)


class Chunk(Contract):
    source_path: str
    title: str
    page_number: int = Field(ge=1)
    chunk_ordinal: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    extraction_method: Literal["native", "ocr"]
    text: str


class Evidence(Chunk):
    chunk_id: str
    document_id: str
    content_sha256: str
    rank: int = Field(ge=1)
    score: float
    label: str = ""


class FileResult(Contract):
    source_path: str
    status: Literal["indexed", "unchanged", "duplicate", "failed"]
    document_id: str = ""
    duplicate_of: str | None = None
    pages: int = 0
    chunks: int = 0
    error: str | None = None


class IngestReport(Contract):
    files: list[FileResult] = Field(default_factory=list)
    maintenance_error: str | None = None
    elapsed_seconds: float = 0

    @property
    def failed(self) -> bool:
        return bool(self.maintenance_error or any(f.status == "failed" for f in self.files))


class CitedText(Contract):
    text: str
    citations: list[str]


class PaperBreakdown(Contract):
    document_id: str
    problem: str
    method: str
    findings: str
    limitations: str
    citations: list[str]


class Breakdown(Contract):
    subquestions: list[str] = Field(max_length=4)
    papers: list[PaperBreakdown]
    gaps: list[str]


class Hypothesis(Contract):
    text: str
    suggested_experiment: str
    citations: list[str] = Field(default_factory=list)


class ResearchPlan(Contract):
    follow_up_queries: list[str] = Field(max_length=4)
    working_findings: list[CitedText]
    hypotheses: list[Hypothesis]


class ResearchReport(Contract):
    findings: list[CitedText]
    agreements_conflicts: list[CitedText]
    remaining_gaps: list[str]
    hypotheses: list[Hypothesis]


class Claim(CitedText):
    claim_id: str = Field(min_length=1)


class AnswerSection(Contract):
    heading: str
    claims: list[Claim]


class DraftAnswer(Contract):
    sections: list[AnswerSection]
    hypotheses: list[Hypothesis]
    limitations: list[str]


class EvidenceQuote(Contract):
    citation: str
    quote: str


class ClaimCheck(Contract):
    claim_id: str
    verdict: Literal["supported", "contradicted", "insufficient"]
    reason: str
    evidence_quotes: list[EvidenceQuote]
    claim_text: str = ""


class Verification(Contract):
    claim_checks: list[ClaimCheck]


class StageEvent(Contract):
    stage: str
    message: str


class AnswerResult(Contract):
    status: Literal["ok", "insufficient_evidence"]
    question: str
    answer_markdown: str
    breakdown: Breakdown | None = None
    research: ResearchReport | None = None
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    sources: list[Evidence] = Field(default_factory=list)
    models: dict[str, str | None] = Field(default_factory=dict)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
