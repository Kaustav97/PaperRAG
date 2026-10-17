"""Exercise saved-answer compatibility through the actual Streamlit surface."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paperrag import config
from paperrag.config import Settings
from pydantic import SecretStr
from paperrag.llm import LLM
from paperrag.models import AnswerResult, ClaimCheck
from paperrag.pipeline import INSUFFICIENT, Pipeline
from paperrag.ingest import Ingestor
from paperrag.storage import Store, file_hash
from test_ingest import TinyEmbedder, _tokenizer, _write_pdf


APP = Path(__file__).resolve().parents[1] / "src" / "paperrag" / "web.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(data_dir=tmp_path / "data", pdf_dir=tmp_path / "papers")
    settings.pdf_dir.mkdir()
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


def library_settings(folder):
    return Settings(pdf_dir=folder, data_dir=folder / ".paperrag")


def test_default_empty_folder_owns_storage(app, tmp_path):
    app, _ = app
    assert [metric.value for metric in app.metric] == ["0", "0", "0", "0"]
    assert (tmp_path / "papers" / ".paperrag" / "manifest.sqlite3").is_file()
    assert not (tmp_path / "data").exists()
    assert any(f"Shared index: {tmp_path / 'papers' / '.paperrag'}" == c.value for c in app.caption)


def test_ui_ingestion_writes_only_active_library(app, tmp_path, monkeypatch):
    app, _ = app
    folder = tmp_path / "chosen"
    pdf = folder / "chosen-paper.pdf"
    _write_pdf(pdf, [["Distinct chosen paper evidence with native PDF text."]])
    original_hash = file_hash(pdf)
    # Only inject deterministic local models; ingestion and storage remain real.
    monkeypatch.setattr("paperrag.ingest.Ingestor", lambda settings, **kwargs: Ingestor(
        settings, embedder=TinyEmbedder(), tokenizer=_tokenizer(), **kwargs))
    app.text_input(key="pdf-directory").set_value(str(folder)).run()
    next(b for b in app.button if b.label == "Ingest directory").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["ingest_report"]["files"][0]["status"] == "indexed"
    next(b for b in app.button if b.label == "Refresh library").click().run()
    assert any(c.label.startswith("chosen-paper") for c in app.checkbox)
    assert app.metric[0].value == "1"
    settings = library_settings(folder)
    with Store(settings) as store:
        assert [d["source_path"] for d in store.documents()] == [str(pdf)]
        rows = store.table.to_arrow().to_pylist()
        assert {row["source_path"] for row in rows} == {str(pdf)}
        assert any("Distinct chosen paper" in row["text"] for row in rows)
    assert list(settings.data_dir.glob("cache/**/pages.jsonl"))
    assert list(settings.data_dir.glob("cache/**/*.arrow"))
    assert (settings.data_dir / "lancedb" / "chunks.lance").is_dir()
    assert file_hash(pdf) == original_hash
    assert not (tmp_path / "data").exists()


def test_switching_libraries_clears_only_cross_library_state(app, tmp_path):
    app, _ = app
    folders = [tmp_path / "folder_a", tmp_path / "folder_b"]
    for folder, title in zip(folders, ("alpha", "beta")):
        pdf = folder / f"{title}.pdf"
        _write_pdf(pdf, [[f"{title} unique evidence for this library."]])
        report = Ingestor(library_settings(folder), embedder=TinyEmbedder(),
                          tokenizer=_tokenizer()).ingest_paths([folder])
        assert report.files[0].status == "indexed"
    a, b = folders
    app.text_input(key="pdf-directory").set_value(str(a)).run()
    assert any(c.label.startswith("alpha") for c in app.checkbox)
    select = next(c for c in app.checkbox if c.key.startswith("select-"))
    select.check().run()
    old_select = select.key
    app.session_state["query_result"] = saved_answer()
    app.session_state["query_error"] = "old research error"
    app.session_state["preview_chunk"] = "old-preview"
    app.session_state["view-old"] = True
    app.session_state["ingest_report"] = {"files": [], "maintenance_error": "old ingest error"}
    app.text_area(key="research-question").set_value("Old question")
    app.radio(key="research-scope").set_value("Selected papers")
    app.run()
    assert len(app.download_button) == 1
    # A syntactically different path with the same canonical value is not a switch.
    app.text_input(key="pdf-directory").set_value(str(a) + "/.").run()
    assert app.session_state["query_result"] == saved_answer()
    assert app.session_state["selected_documents"]
    # Stale pagination must be reset before the smaller library's widget is built.
    app.session_state["library-page"] = 9
    app.text_input(key="pdf-directory").set_value(str(b)).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert any(c.label.startswith("beta") for c in app.checkbox)
    assert not any(c.label.startswith("alpha") for c in app.checkbox)
    assert not app.download_button
    assert not app.error
    assert not app.dataframe
    assert app.session_state["selected_documents"] == []
    for key in ("query_result", "query_error", "preview_chunk", "ingest_report", old_select, "view-old"):
        assert key not in app.session_state
    assert app.text_area(key="research-question").value == ""
    assert app.radio(key="research-scope").value == "All indexed papers"
    assert app.number_input(key="library-page").value == 1
    app.text_input(key="pdf-directory").set_value(str(a)).run()
    assert any(c.label.startswith("alpha") for c in app.checkbox)
    assert not any(c.label.startswith("beta") for c in app.checkbox)
    for folder, title in zip(folders, ("alpha", "beta")):
        with Store(library_settings(folder)) as store:
            assert [d["title"] for d in store.documents()] == [title]
            assert {r["source_path"] for r in store.table.to_arrow().to_pylist()} == {str(folder / f"{title}.pdf")}


@pytest.mark.parametrize("invalid", ["blank", "missing", "file", "storage"])
def test_invalid_folder_stops_before_stale_results_or_fallback(app, tmp_path, invalid):
    app, _ = app
    project_data = tmp_path / "data"
    project_data.mkdir()
    sentinel = project_data / "sentinel"
    sentinel.write_bytes(b"existing project data")
    prior = tmp_path / "papers"
    app.session_state["query_result"] = saved_answer()
    app.run()
    assert app.download_button
    target = tmp_path / invalid
    if invalid == "file":
        target.write_text("not a directory")
    elif invalid == "storage":
        target.mkdir()
        (target / ".paperrag").write_text("conflicting file")
    value = "  " if invalid == "blank" else str(target)
    app.text_input(key="pdf-directory").set_value(value).run()
    assert not app.exception
    assert app.error
    assert not app.metric
    assert not app.download_button
    assert "A complete answer." not in [m.value for m in app.markdown]
    assert sentinel.read_bytes() == b"existing project data"
    assert not (project_data / "manifest.sqlite3").exists()
    if invalid == "missing":
        assert not target.exists()
    elif invalid == "storage":
        assert str(target / ".paperrag") in app.error[0].value
        assert (target / ".paperrag").read_text() == "conflicting file"
    app.text_input(key="pdf-directory").set_value(str(prior)).run()
    assert not app.exception
    assert app.metric[0].value == "0"
    if invalid != "storage":
        assert app.session_state["query_result"] == saved_answer()


def test_model_controls_use_configuration_and_survive_library_switch(app, tmp_path, monkeypatch):
    app, calls = app
    settings = Settings(
        pdf_dir=tmp_path / "papers", base_url="http://127.0.0.1:9/v1",
        api_key=SecretStr("local-test-key"),
        models=config.Models(**{
            stage: config.StageModel(model=f"configured-{stage}", options={"temperature": 0.2})
            for stage in ("breakdown", "research", "synthesis")
        }),
    )
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    # Start a fresh session so widget defaults come from configuration.
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    for stage in ("breakdown", "research", "synthesis"):
        assert app.text_input(key=f"model-{stage}").value == f"configured-{stage}"
        assert json.loads(app.text_area(key=f"model-options-{stage}").value) == {"temperature": 0.2}
    saved = saved_answer()
    app.session_state["query_result"] = saved
    options = {"extra_body": {"plugins": [{"id": "fusion", "preset": "general-budget"}]}}
    app.text_input(key="model-research").set_value("openrouter/fusion")
    app.text_area(key="model-options-research").set_value(json.dumps(options)).run()
    assert not app.exception
    assert not app.button(key="research-answer").disabled
    assert app.session_state["query_result"] == saved
    app.download_button[0].click().run()
    assert app.session_state["query_result"] == saved
    folder = tmp_path / "second-library"
    folder.mkdir()
    app.text_input(key="pdf-directory").set_value(str(folder)).run()
    assert app.text_input(key="model-research").value == "openrouter/fusion"
    assert json.loads(app.text_area(key="model-options-research").value) == options
    assert app.text_input(key="model-synthesis").value == "configured-synthesis"
    assert calls == []


@pytest.mark.parametrize("invalid", [
    "{", "[]", "null", '{"model": "override"}',
    '{"extra_body": {"messages": []}}', '{"stream": true}',
    '{"extra_body": {"api_key": "override"}}',
])
def test_invalid_model_options_block_answers_not_saved_results_or_search(app, tmp_path, monkeypatch, invalid):
    app, calls = app
    settings = Settings(
        pdf_dir=tmp_path / "papers", base_url="http://127.0.0.1:9/v1",
        api_key=SecretStr("local-test-key"),
        models=config.Models(**{
            stage: config.StageModel(model="local-test-model")
            for stage in ("breakdown", "research", "synthesis")
        }),
    )
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    app.run()
    for stage in ("breakdown", "research", "synthesis"):
        app.text_input(key=f"model-{stage}").set_value("local-test-model")
    saved = saved_answer()
    app.session_state["query_result"] = saved
    app.text_area(key="model-options-research").set_value(invalid).run()
    assert not app.exception
    assert any("Research model settings:" in e.value for e in app.error)
    assert app.button(key="research-answer").disabled
    assert not app.button(key="research-search").disabled
    assert app.session_state["query_result"] == saved
    app.download_button[0].click().run()
    assert app.session_state["query_result"] == saved
    app.text_area(key="model-options-research").set_value("{}").run()
    assert not app.error
    assert not app.button(key="research-answer").disabled
    assert app.session_state["query_result"] == saved
    app.text_input(key="model-synthesis").set_value("   ").run()
    assert app.button(key="research-answer").disabled
    assert any("PAPERRAG_SYNTHESIS_MODEL" in warning.value for warning in app.warning)
    assert calls == []
