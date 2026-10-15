"""Exercise saved-answer compatibility through the actual Streamlit surface."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paperrag import config
from paperrag.config import Settings
from paperrag.llm import LLM
from paperrag.models import AnswerResult, ClaimCheck
from paperrag.pipeline import INSUFFICIENT, Pipeline


APP = Path(__file__).resolve().parents[1] / "src" / "paperrag" / "web.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_dir=tmp_path / "data", pdf_dir=tmp_path / "papers")
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    calls = []

    def unexpected_call(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Rendering saved results must not call the provider or rerun research")

    monkeypatch.setattr(LLM, "complete_json", unexpected_call)
    monkeypatch.setattr(Pipeline, "answer", unexpected_call)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    yield app, calls
    assert calls == []


def test_legacy_answer_is_cleared_without_paid_rerun(app):
    app, calls = app
    value = AnswerResult(
        status="ok", question="An old question", answer_markdown="An old answer.",
        claim_checks=[ClaimCheck(claim_id="c1", verdict="supported", role="answer",
                                 reason="Legacy fixture", evidence_quotes=[])],
    ).model_dump()
    del value["claim_checks"][0]["role"]
    app.session_state["query_result"] = {"kind": "answer", "value": value}
    app.session_state["preview_chunk"] = "old-preview"
    app.run()

    assert not app.exception
    assert "query_result" not in app.session_state
    assert "preview_chunk" not in app.session_state
    assert any(info.value == "Previous answer cleared because it lacks question-relevance checks. Submit the question again."
               for info in app.info)
    assert "An old answer." not in [item.value for item in app.markdown]
    assert len(app.download_button) == 0
    assert calls == []


@pytest.mark.parametrize("kind", ["answer", "search"])
def test_empty_check_answers_and_search_results_are_preserved(app, kind):
    app, calls = app
    if kind == "answer":
        saved = {"kind": "answer", "value": AnswerResult(
            status="insufficient_evidence", question="An unanswered question", answer_markdown=INSUFFICIENT,
        ).model_dump()}
    else:
        saved = {"kind": "search", "question": "A local search", "sources": []}
    app.session_state["query_result"] = saved
    app.session_state["preview_chunk"] = "existing-preview"
    app.run()

    assert not app.exception
    assert app.session_state["query_result"] == saved
    assert app.session_state["preview_chunk"] == "existing-preview"
    assert not any(info.value.startswith("Previous answer cleared") for info in app.info)
    if kind == "answer":
        assert INSUFFICIENT in [item.value for item in app.markdown]
    else:
        assert any("Retrieval only" in item.value for item in app.caption)
    assert calls == []


def saved_answer() -> dict:
    return {"kind": "answer", "value": AnswerResult(
        status="ok", question="A question", answer_markdown="A complete answer.",
        claim_checks=[ClaimCheck(claim_id="c1", verdict="supported", role="answer",
                                 reason="Reasonable", evidence_quotes=[])],
    ).model_dump()}


def test_saved_answer_offers_save_response_download(app):
    app, calls = app
    app.session_state["query_result"] = saved_answer()
    app.run()

    assert not app.exception
    assert len(app.download_button) == 1
    assert app.download_button[0].proto.label == "Save response"
    assert "A complete answer." in [item.value for item in app.markdown]
    assert calls == []


def test_save_response_click_reruns_without_modifying_state_or_calling(app):
    app, calls = app
    saved = saved_answer()
    app.session_state["query_result"] = saved
    app.session_state["preview_chunk"] = "existing-preview"
    app.run()
    assert len(app.download_button) == 1

    app.download_button[0].click().run()

    assert not app.exception
    assert app.session_state["query_result"] == saved
    assert app.session_state["preview_chunk"] == "existing-preview"
    assert len(app.download_button) == 1
    assert calls == []


def test_search_only_result_has_no_save_response(app):
    app, calls = app
    app.session_state["query_result"] = {"kind": "search", "question": "A local search", "sources": []}
    app.run()

    assert not app.exception
    assert len(app.download_button) == 0
    assert calls == []
