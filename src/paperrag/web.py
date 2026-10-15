"""Streamlit surface over the same services as CLI. Paid calls occur only on submit."""
from pathlib import Path

import streamlit as st

from paperrag.config import load_settings
from paperrag.ingest import Ingestor
from paperrag.models import AnswerResult, Evidence
from paperrag.pipeline import Pipeline
from paperrag.response_export import render_response_markdown
from paperrag.retrieval import RetrievalSession
from paperrag.sources import render_page, source_path
from paperrag.storage import Store

st.set_page_config(page_title="PaperRAG", page_icon=None, layout="wide")
st.title("PaperRAG")
st.caption("Local papers · physical-page evidence · bounded research")
settings = load_settings()
st.caption(f"Shared index: {settings.data_dir}")
st.info("Local parsing, OCR and search stay on this machine. Research answer sends retrieved excerpts and metadata to your configured endpoint; hosted endpoints receive that data.")


def source_cards(sources):
    st.subheader("Sources")
    for number, evidence in enumerate(sources, 1):
        label = evidence.label or f"S{number}"
        with st.expander(f"[{label}] {evidence.title} — physical page {evidence.page_number}"):
            st.text(evidence.source_path)
            st.caption(f"{evidence.extraction_method} · characters {evidence.char_start}–{evidence.char_end} · {evidence.chunk_id[:12]}")
            st.text(evidence.text)
            # Only preview intent is persisted. Re-rendering is local and rechecks the source hash.
            if st.button(f"View page [{label}]", key="view-" + evidence.chunk_id):
                st.session_state["preview_chunk"] = evidence.chunk_id
            if st.session_state.get("preview_chunk") == evidence.chunk_id:
                try:
                    path, png = render_page(settings, evidence)
                    st.image(png, caption=f"{evidence.title} — physical page {evidence.page_number}", width=740)
                    # Hash is checked again at the moment download bytes are prepared.
                    path = source_path(settings, evidence)
                    with path.open("rb") as pdf:
                        st.download_button("Download original PDF", pdf, file_name=path.name,
                                           mime="application/pdf", key="download-" + evidence.chunk_id)
                except (ValueError, OSError) as exc:
                    st.warning(str(exc))
            else:
                try:
                    source_path(settings, evidence)
                except (ValueError, OSError) as exc:
                    st.warning(str(exc))


library, research_tab = st.tabs(["Library", "Research"])
with library:
    st.subheader("Committed library")
    if st.button("Refresh library"):
        st.rerun()
    with Store(settings) as store:
        stats = store.stats()
        cols = st.columns(4)
        for col, key in zip(cols, ("documents", "pages", "chunks", "empty_pages")):
            col.metric(key.replace("_", " ").title(), stats[key])
        if stats["pending"]:
            st.warning("An interrupted commit is pending reconciliation. Run ingest or paperrag index to recover it.")
        with st.expander("Index readiness"):
            st.json(stats["indexes"])
        page = st.number_input("Library page", min_value=1, max_value=max(1, (stats["documents"] + 24) // 25), value=1, step=1)
        documents = store.documents(offset=(page - 1) * 25)
    selected = set(st.session_state.get("selected_documents", []))
    for document in documents:
        doc_id = document["document_id"]
        checked = st.checkbox(f"{document['title']} · {document['pages']} pages · {document['chunks']} chunks",
                              value=doc_id in selected, key="select-" + doc_id)
        if checked:
            selected.add(doc_id)
        else:
            selected.discard(doc_id)
        st.caption(document["source_path"] + " · ID " + doc_id)
    st.session_state["selected_documents"] = sorted(selected)
    st.caption(f"{len(selected)} selected across library pages")
    with st.form("ingest"):
        directory = st.text_input("Local PDF directory", value=str(settings.pdf_dir))
        force_ocr = st.checkbox("Force local OCR on every page")
        submitted = st.form_submit_button("Ingest directory")
    if submitted:
        try:
            with st.status("Indexing local PDFs", expanded=True) as status:
                live = st.empty()
                report = Ingestor(settings, on_progress=lambda event: live.text(event.message)).ingest_paths([Path(directory)], force_ocr=force_ocr)
                status.update(label="Indexing completed with failures" if report.failed else "Index ready", state="error" if report.failed else "complete")
            st.session_state["ingest_report"] = report.model_dump()
        except Exception as exc:
            st.error(str(exc))
    if "ingest_report" in st.session_state:
        report = st.session_state["ingest_report"]
        st.dataframe(report["files"], hide_index=True)
        if report["maintenance_error"]:
            st.error(report["maintenance_error"])
        st.caption("Refresh library to see the latest committed counts.")

with research_tab:
    st.subheader("Research over indexed evidence")
    st.caption("1. Local index → 2. Deconstruction → 3. Bounded research → 4. Synthesis + independent excerpt verification")
    try:
        settings.require_llm()
        configured = True
    except ValueError as exc:
        configured = False
        st.warning(str(exc))
    question = st.text_area("Research question", placeholder="What non-parametric memory does the RAG model use?", key="research-question")
    scope = st.radio("Corpus scope", ["All indexed papers", "Selected papers"], horizontal=True, key="research-scope")
    action_row = st.columns(3)
    search_clicked = action_row[0].button("Search excerpts", key="research-search")
    answer_clicked = action_row[1].button("Research answer", key="research-answer")
    save_action = action_row[2]
    if search_clicked or answer_clicked:
        try:
            ids = st.session_state["selected_documents"] if scope == "Selected papers" else None
            if ids == []:
                raise ValueError("Select at least one paper in Library or choose All indexed papers.")
            if answer_clicked:
                settings.require_llm()
                with st.status("Research in progress", expanded=True) as status:
                    def update(event):
                        st.write(f"{event.stage}: {event.message}")
                    result = Pipeline(settings).answer(question, document_ids=ids, on_progress=update)
                    status.update(label="Checked against retrieved paper excerpts", state="complete")
                st.session_state["query_result"] = {"kind": "answer", "value": result.model_dump()}
            else:
                with st.spinner("Searching local excerpts"):
                    with RetrievalSession(settings, document_ids=ids) as session:
                        sources = session.retrieve(question)
                st.session_state["query_result"] = {"kind": "search", "question": question,
                    "sources": [e.model_copy(update={"label": f"S{i}"}).model_dump() for i, e in enumerate(sources, 1)]}
            st.session_state.pop("query_error", None)
            st.session_state.pop("preview_chunk", None)
        except Exception as exc:
            st.session_state["query_error"] = str(exc)
            st.session_state.pop("query_result", None)
    if st.session_state.get("query_error"):
        st.error(st.session_state["query_error"])
    saved = st.session_state.get("query_result")
    if saved and saved["kind"] == "answer" and any("role" not in check for check in saved["value"].get("claim_checks", [])):
        st.session_state.pop("query_result", None)
        st.session_state.pop("preview_chunk", None)
        saved = None
        st.info("Previous answer cleared because it lacks question-relevance checks. Submit the question again.")
    if saved and saved["kind"] == "answer":
        result = AnswerResult.model_validate(saved["value"])
        save_action.download_button(
            "Save response",
            data=render_response_markdown(result),
            file_name="paperrag-research-response.md",
            mime="text/markdown",
            key="save-response",
        )
        st.caption("Result question: " + result.question)
        st.markdown(result.answer_markdown)
        st.caption("Checked against retrieved paper excerpts — evidence support, not independent scientific truth.")
        if result.status == "insufficient_evidence":
            if result.hypotheses:
                st.subheader("Hypotheses — not established by the cited papers")
                for hypothesis in result.hypotheses:
                    st.write(hypothesis.text)
                    st.write("Suggested experiment: " + hypothesis.suggested_experiment)
            if result.limitations:
                st.subheader("Limitations")
                for limitation in result.limitations:
                    st.write(limitation)
        with st.expander("Structured deconstruction and research"):
            st.json({"breakdown": result.breakdown.model_dump() if result.breakdown else None,
                     "research": result.research.model_dump() if result.research else None})
        st.subheader("Fact checks — rejected claims are not answer facts")
        st.dataframe([c.model_dump() for c in result.claim_checks], hide_index=True)
        source_cards(result.sources)
    elif saved:
        st.caption("Result question: " + saved["question"])
        st.caption("Retrieval only. Nearest neighbors are not proof of relevance.")
        sources = [Evidence.model_validate(e) for e in saved["sources"]]
        if not sources:
            st.info("No evidence found. Ingest papers from Library or with paperrag ingest PATH.")
        source_cards(sources)
