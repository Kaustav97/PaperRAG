"""Deterministic consumer-boundary regressions for the research pipeline."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from paperrag.config import Models, Settings, StageModel
from paperrag.llm import LLM, StageError
from paperrag.models import (
    AnswerSection,
    Breakdown,
    Claim,
    ClaimCheck,
    CitedText,
    DraftAnswer,
    Evidence,
    EvidenceQuote,
    Hypothesis,
    PaperBreakdown,
    ResearchPlan,
    ResearchReport,
    Verification,
)
from paperrag.pipeline import INSUFFICIENT, Pipeline


DOC_A = "a" * 64
DOC_B = "b" * 64
UNKNOWN_DOC = "f" * 64


class ScriptedLLM:
    """Test-only model fixture; every call is explicit and recorded."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete_json(self, stage, messages, output_type):
        self.calls.append((stage, messages, output_type))
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


class SessionSpy:
    """A retrieval-session seam that still exercises Pipeline's real expansion loop."""

    def __init__(self, settings, *, embedder=None, document_ids=None, initial=(), followups=None, registry=None):
        self.document_ids = document_ids
        self.calls = []
        self.initial = list(initial)
        self.followups = {key: list(value) for key, value in (followups or {}).items()}
        if registry is not None:
            registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def retrieve(self, question, *, top_k=12, document_ids=None):
        assert document_ids is None, "Pipeline must retain constructor scope for expansion queries"
        self.calls.append(question)
        if len(self.calls) == 1:
            rows = list(self.initial)
        else:
            rows = list(self.followups.get(question, self.initial))
        if self.document_ids:
            assert all(item.document_id in set(self.document_ids) for item in rows)
        return rows


def evidence(chunk_id: str, document_id: str = DOC_A, text: str = "The paper reports the observed result.", *, rank: int = 1) -> Evidence:
    return Evidence(
        source_path=f"/papers/{document_id[:4]}.pdf",
        title="A paper",
        page_number=1,
        chunk_ordinal=0,
        char_start=0,
        char_end=len(text),
        extraction_method="native",
        text=text,
        chunk_id=chunk_id,
        document_id=document_id,
        content_sha256="c" * 64,
        rank=rank,
        score=1.0 / rank,
    )


def settings(*, configured: bool = True) -> Settings:
    if not configured:
        return Settings()
    stage = StageModel(model="fixture-model")
    return Settings(
        base_url="https://fixture.invalid/v1",
        api_key=SecretStr("fixture-key"),
        models=Models(breakdown=stage, research=stage, synthesis=stage),
    )


def outputs(*, claim_text="The paper reports the observed result.", verification=None):
    source_label = "S1"
    breakdown = Breakdown(
        subquestions=["What was observed?"],
        papers=[PaperBreakdown(document_id=DOC_A, problem="A problem", method="A method", findings="A finding", limitations="A limit", citations=[source_label])],
        gaps=["not established in retrieved excerpts"],
    )
    plan = ResearchPlan(
        follow_up_queries=[],
        working_findings=[CitedText(text="The working finding.", citations=[source_label])],
        hypotheses=[],
    )
    report = ResearchReport(
        findings=[CitedText(text="The reported finding.", citations=[source_label])],
        agreements_conflicts=[],
        remaining_gaps=[],
        hypotheses=[],
    )
    draft = DraftAnswer(
        sections=[AnswerSection(heading="Findings", claims=[Claim(claim_id="c1", text=claim_text, citations=[source_label])])],
        hypotheses=[],
        limitations=[],
    )
    check = verification or Verification(
        claim_checks=[ClaimCheck(claim_id="c1", verdict="supported", role="answer", reason="The excerpt states it.", evidence_quotes=[EvidenceQuote(citation=source_label, quote=claim_text)])]
    )
    return [breakdown, plan, report, draft, check]


def pipeline_with(initial, llm, *, configured=True, document_ids=None, followups=None, sessions=None):
    def factory(settings, *, embedder=None, document_ids=None):
        return SessionSpy(settings, embedder=embedder, document_ids=document_ids, initial=initial, followups=followups, registry=sessions)

    return Pipeline(settings(configured=configured), llm=llm, session_factory=factory), factory


def test_pipeline_bounded_expansion_preserves_scope_labels_and_budgets():
    first = evidence("1" * 64)
    bulk = [
        evidence(
            f"{i + 100:064x}",
            document_id=DOC_A if i % 2 == 0 else DOC_B,
            text="x" * 2000,
            rank=i + 2,
        ).model_copy(update={"page_number": i + 2})
        for i in range(19)
    ]
    second = evidence("2" * 64, document_id=DOC_B, text="Follow-up evidence.")
    llm = ScriptedLLM(outputs())
    sessions = []
    pipeline, _ = pipeline_with([first, *bulk], llm, document_ids=[DOC_A, DOC_B], followups={"follow-up": [second], "second": [second]}, sessions=sessions)

    # Duplicate/case-folded and blank queries must become one bounded local round.
    llm.outputs[1] = ResearchPlan(
        follow_up_queries=[" follow-up ", "FOLLOW-UP", "", "second"],
        working_findings=[CitedText(text="The working finding.", citations=["S1"])],
        hypotheses=[],
    )
    result = pipeline.answer("question", document_ids=[DOC_A, DOC_B])

    assert result.status == "ok"
    assert len(sessions) == 1
    assert sessions[0].document_ids == [DOC_A, DOC_B]
    assert sessions[0].calls == ["question", "follow-up", "second"]
    assert len(result.sources) <= 16
    assert [source.label for source in result.sources] == [f"S{i}" for i in range(1, len(result.sources) + 1)]
    for _stage, messages, _contract in llm.calls:
        payload = json.loads(messages[-1]["content"])
        supplied = payload["quoted_sources"]
        assert len(supplied) <= 16
        assert sum(len(source["text"]) for source in supplied) <= 24_000
        assert [source["label"] for source in supplied] == [f"S{i}" for i in range(1, len(supplied) + 1)]


def test_pipeline_rejects_fabricated_references_at_each_stage():
    first = evidence("1" * 64)
    cases = []

    bad_breakdown = outputs()
    bad_breakdown[0] = Breakdown(
        subquestions=[],
        papers=[PaperBreakdown(document_id=UNKNOWN_DOC, problem="", method="", findings="", limitations="", citations=[])],
        gaps=[],
    )
    cases.append(("breakdown", bad_breakdown))

    bad_plan = outputs()
    bad_plan[1] = ResearchPlan(follow_up_queries=[], working_findings=[CitedText(text="bad", citations=["S99"])], hypotheses=[])
    cases.append(("research", bad_plan))

    bad_report = outputs()
    bad_report[2] = ResearchReport(findings=[CitedText(text="bad", citations=["S99"])], agreements_conflicts=[], remaining_gaps=[], hypotheses=[])
    cases.append(("research", bad_report))

    bad_draft = outputs()
    bad_draft[3] = DraftAnswer(sections=[AnswerSection(heading="", claims=[Claim(claim_id="c1", text="bad", citations=["S99"])])], hypotheses=[], limitations=[])
    cases.append(("synthesis", bad_draft))

    bad_verification = outputs()
    bad_verification[4] = Verification(claim_checks=[ClaimCheck(claim_id="c1", verdict="supported", role="answer", reason="", evidence_quotes=[EvidenceQuote(citation="S99", quote="bad")])])
    cases.append(("synthesis", bad_verification))

    for expected_stage, scripted in cases:
        with pytest.raises(StageError, match="Unknown (document ID|citation label)") as caught:
            pipeline, _ = pipeline_with([first], ScriptedLLM(scripted))
            pipeline.answer("question")
        assert caught.value.stage == expected_stage


def test_supported_quote_must_exist_and_bad_supported_quote_is_insufficient():
    source_text = "The observed result is reproducible."
    first = evidence("1" * 64, text=source_text)
    draft = DraftAnswer(
        sections=[AnswerSection(heading="Findings", claims=[
            Claim(claim_id="valid", text="The observed result is reproducible.", citations=["S1"]),
            Claim(claim_id="bad", text="The hidden result is definitive.", citations=["S1"]),
        ])],
        hypotheses=[], limitations=[],
    )
    verification = Verification(claim_checks=[
        ClaimCheck(claim_id="valid", verdict="supported", role="answer", reason="quoted", evidence_quotes=[EvidenceQuote(citation="S1", quote=source_text)]),
        ClaimCheck(claim_id="bad", verdict="supported", role="answer", reason="not actually quoted", evidence_quotes=[EvidenceQuote(citation="S1", quote="not in the excerpt")]),
    ])
    scripted = outputs()
    scripted[3], scripted[4] = draft, verification
    result = pipeline_with([first], ScriptedLLM(scripted))[0].answer("question")

    assert result.status == "ok"
    checks = {check.claim_id: check for check in result.claim_checks}
    assert checks["valid"].verdict == "supported"
    assert checks["bad"].verdict == "insufficient"
    assert checks["bad"].claim_text == "The hidden result is definitive."
    assert "The observed result is reproducible. [S1]" in result.answer_markdown
    assert "The hidden result is definitive." not in result.answer_markdown


def test_supported_context_and_disclaimer_abstain_without_losing_brainstorming():
    context = "The paper uses dense retrieval for language tasks."
    disclaimer = "The excerpts do not establish a pediatric antibiotic dose."
    draft = DraftAnswer(
        sections=[AnswerSection(heading="Findings", claims=[
            Claim(claim_id="context", text=context, citations=["S1"]),
            Claim(claim_id="disclaimer", text=disclaimer, citations=["S1"]),
        ])],
        hypotheses=[Hypothesis(text="Investigate clinical evidence separately.", suggested_experiment="Review relevant clinical studies.")],
        limitations=["Only NLP papers were indexed."],
    )
    verification = Verification(claim_checks=[
        ClaimCheck(claim_id="context", verdict="supported", role="context", reason="Background only.",
                   evidence_quotes=[EvidenceQuote(citation="S1", quote=context)]),
        ClaimCheck(claim_id="disclaimer", verdict="supported", role="insufficiency", reason="No requested dose.",
                   evidence_quotes=[EvidenceQuote(citation="S1", quote=disclaimer)]),
    ])
    scripted = outputs()
    scripted[3], scripted[4] = draft, verification
    llm = ScriptedLLM(scripted)
    result = pipeline_with([evidence("1" * 64, text=context + " " + disclaimer)], llm)[0].answer(
        "What antibiotic dose should be prescribed to a child?"
    )

    assert (result.status, result.answer_markdown) == ("insufficient_evidence", INSUFFICIENT)
    assert [(c.role, c.verdict, c.claim_text) for c in result.claim_checks] == [
        ("context", "supported", context), ("insufficiency", "supported", disclaimer),
    ]
    assert result.hypotheses == draft.hypotheses
    assert result.limitations == draft.limitations
    assert [call[0] for call in llm.calls] == ["breakdown", "research", "research", "synthesis", "synthesis"]


def test_responsive_answer_excludes_supported_background():
    answer = "RAG uses a dense vector index of Wikipedia as non-parametric memory."
    context = "The paper also discusses language generation."
    draft = DraftAnswer(
        sections=[AnswerSection(heading="Findings", claims=[
            Claim(claim_id="answer", text=answer, citations=["S1"]),
            Claim(claim_id="context", text=context, citations=["S1"]),
        ])],
        hypotheses=[], limitations=[],
    )
    verification = Verification(claim_checks=[
        ClaimCheck(claim_id=claim_id, verdict="supported", role=role, reason="Quoted.",
                   evidence_quotes=[EvidenceQuote(citation="S1", quote=text)])
        for claim_id, role, text in [("answer", "answer", answer), ("context", "context", context)]
    ])
    scripted = outputs()
    scripted[3], scripted[4] = draft, verification
    result = pipeline_with([evidence("1" * 64, text=answer + " " + context)], ScriptedLLM(scripted))[0].answer(
        "What non-parametric memory does RAG use?"
    )

    assert (result.status, result.answer_markdown) == ("ok", answer + " [S1]")
    assert [c.verdict for c in result.claim_checks] == ["supported", "supported"]


@pytest.mark.parametrize("question, text", [
    ("Did the intervention improve accuracy?", "The intervention did not improve accuracy."),
    ("Do the excerpts establish a pediatric dose?", "The excerpts do not establish a pediatric dose."),
])
def test_responsive_negative_and_excerpt_coverage_answers_remain_eligible(question, text):
    result = pipeline_with([evidence("1" * 64, text=text)], ScriptedLLM(outputs(claim_text=text)))[0].answer(question)
    assert (result.status, result.answer_markdown) == ("ok", text + " [S1]")
    assert result.claim_checks[0].role == "answer"


def test_answer_role_does_not_bypass_quote_validation():
    scripted = outputs(claim_text="A claim absent from the excerpt.")
    result = pipeline_with([evidence("1" * 64)], ScriptedLLM(scripted))[0].answer("What was observed?")
    assert (result.status, result.answer_markdown) == ("insufficient_evidence", INSUFFICIENT)
    assert result.claim_checks[0].role == "answer"
    assert result.claim_checks[0].verdict == "insufficient"
    assert result.claim_checks[0].evidence_quotes == []


def test_contradicted_and_insufficient_claims_are_not_emitted_and_all_unsupported_is_exact():
    first = evidence("1" * 64, text="The excerpt contradicts claim one and says nothing about claim two.")
    draft = DraftAnswer(
        sections=[AnswerSection(heading="Findings", claims=[
            Claim(claim_id="contradicted", text="Claim one is supported.", citations=["S1"]),
            Claim(claim_id="insufficient", text="Claim two is supported.", citations=["S1"]),
        ])],
        hypotheses=[Hypothesis(text="Try a new experiment.", suggested_experiment="Run it locally.")],
        limitations=["Only one excerpt was retrieved."],
    )
    verification = Verification(claim_checks=[
        ClaimCheck(claim_id="contradicted", verdict="contradicted", role="answer", reason="The excerpt disagrees.", evidence_quotes=[EvidenceQuote(citation="S1", quote="The excerpt contradicts claim one")]),
        ClaimCheck(claim_id="insufficient", verdict="insufficient", role="answer", reason="No support.", evidence_quotes=[]),
    ])
    scripted = outputs()
    scripted[3], scripted[4] = draft, verification
    result = pipeline_with([first], ScriptedLLM(scripted))[0].answer("question")

    assert result.status == "insufficient_evidence"
    assert result.answer_markdown == INSUFFICIENT
    assert "Claim one is supported" not in result.answer_markdown
    assert "Claim two is supported" not in result.answer_markdown
    assert [check.claim_text for check in result.claim_checks] == ["Claim one is supported.", "Claim two is supported."]
    assert result.hypotheses == draft.hypotheses
    assert result.limitations == draft.limitations


@pytest.mark.parametrize("kind", ["duplicate", "missing"])
def test_duplicate_or_missing_claim_checks_fail_after_corrective_retry(kind):
    first = evidence("1" * 64)
    draft = DraftAnswer(
        sections=[AnswerSection(heading="", claims=[
            Claim(claim_id="c1", text="One.", citations=["S1"]),
            Claim(claim_id="c2", text="Two.", citations=["S1"]),
        ])], hypotheses=[], limitations=[],
    )
    checks = [ClaimCheck(claim_id="c1", verdict="insufficient", role="answer", reason="", evidence_quotes=[])]
    if kind == "duplicate":
        checks.append(checks[0])
    verification = Verification(claim_checks=checks)
    scripted = outputs()
    scripted[3], scripted[4] = draft, verification
    # The single corrective retry also receives the same inconsistent output, so the
    # invariant still hard-fails instead of silently accepting a partial check set.
    scripted.append(verification)

    with pytest.raises(StageError, match="Claim checks must cover") as caught:
        pipeline_with([first], ScriptedLLM(scripted))[0].answer("question")
    assert caught.value.stage == "synthesis verification"


def test_checker_id_mismatch_recovers_on_corrective_retry():
    first = evidence("1" * 64)
    draft = DraftAnswer(
        sections=[AnswerSection(heading="", claims=[
            Claim(claim_id="c1", text="One.", citations=["S1"]),
            Claim(claim_id="c2", text="Two.", citations=["S1"]),
        ])], hypotheses=[], limitations=[],
    )
    bad = Verification(claim_checks=[ClaimCheck(claim_id="c1", verdict="insufficient", role="answer", reason="", evidence_quotes=[])])
    good = Verification(claim_checks=[
        ClaimCheck(claim_id="c1", verdict="insufficient", role="answer", reason="", evidence_quotes=[]),
        ClaimCheck(claim_id="c2", verdict="insufficient", role="answer", reason="", evidence_quotes=[]),
    ])
    scripted = outputs()
    scripted[3], scripted[4] = draft, bad
    scripted.append(good)

    result = pipeline_with([first], ScriptedLLM(scripted))[0].answer("question")

    assert result.status == "insufficient_evidence"
    assert result.answer_markdown == INSUFFICIENT
    assert [c.claim_id for c in result.claim_checks] == ["c1", "c2"]


def test_provider_stage_failure_is_not_insufficient_evidence():
    pipeline, _ = pipeline_with([evidence("1" * 64)], ScriptedLLM([outputs()[0], StageError("research", "provider failed")]))
    with pytest.raises(StageError, match="provider failed") as caught:
        pipeline.answer("question")
    assert caught.value.stage == "research"


def test_empty_corpus_returns_exact_insufficient_without_provider_call():
    llm = ScriptedLLM([])
    pipeline, _ = pipeline_with([], llm, configured=False)
    result = pipeline.answer("question")
    assert result.status == "insufficient_evidence"
    assert result.answer_markdown == INSUFFICIENT
    assert llm.calls == []


def test_missing_configuration_stops_before_provider_request_but_retrieves_locally():
    llm = ScriptedLLM([])
    sessions = []
    pipeline, _ = pipeline_with([evidence("1" * 64)], llm, configured=False, sessions=sessions)
    with pytest.raises(ValueError, match="Research requires explicit configuration"):
        pipeline.answer("question")
    assert sessions[0].calls == ["question"]
    assert llm.calls == []


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(response=response, error=error))


def llm_fixture(response=None, error=None):
    llm = object.__new__(LLM)
    llm.settings = settings()
    llm.client = FakeClient(response=response, error=error)
    return llm


def response(content, *, finish_reason="stop", choices=True):
    rows = [] if not choices else [SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=rows)


def test_complete_json_accepts_one_json_code_fence_and_validates_schema():
    value = Breakdown(subquestions=[], papers=[], gaps=[])
    llm = llm_fixture(response("```json\n" + json.dumps(value.model_dump()) + "\n```"))
    parsed = llm.complete_json("breakdown", [{"role": "user", "content": "x"}], Breakdown)
    assert parsed == value


@pytest.mark.parametrize(
    "fixture, message",
    [
        (response("", choices=False), "Empty or truncated"),
        (response("{}", finish_reason="length"), "Empty or truncated"),
        (response("   "), "empty assistant content"),
        (response("not-json"), "Malformed JSON"),
        (response("{}"), "Malformed JSON"),
    ],
)
def test_complete_json_rejects_empty_truncated_malformed_or_schema_mismatch(fixture, message):
    with pytest.raises(StageError, match=message) as caught:
        llm_fixture(fixture).complete_json("breakdown", [{"role": "user", "content": "x"}], Breakdown)
    assert caught.value.stage == "breakdown"


def test_complete_json_provider_failure_is_named_and_not_synthetic():
    with pytest.raises(StageError, match="Provider call failed") as caught:
        llm_fixture(error=RuntimeError("offline fixture")).complete_json("synthesis", [], DraftAnswer)
    assert caught.value.stage == "synthesis"


@pytest.mark.parametrize("role", [None, "background"])
def test_verification_missing_or_unknown_role_is_a_synthesis_schema_error(role):
    check = outputs()[4].claim_checks[0].model_dump()
    if role is None:
        del check["role"]
    else:
        check["role"] = role
    llm = llm_fixture(response(json.dumps({"claim_checks": [check]})))
    with pytest.raises(StageError, match="schema mismatch") as caught:
        llm.complete_json("synthesis", [{"role": "user", "content": "Verify these claims."}], Verification)
    assert caught.value.stage == "synthesis"
