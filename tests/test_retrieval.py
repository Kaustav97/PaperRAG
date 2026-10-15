from __future__ import annotations

import re
from pathlib import Path

import pytest
import pymupdf

from paperrag.config import Settings
from paperrag.ingest import Ingestor
from paperrag.models import Evidence
from paperrag.retrieval import RetrievalSession, diverse, fuse
from paperrag.sources import UNAVAILABLE, render_page, source_path
from paperrag.storage import document_id, file_hash


class _Encoding:
    def __init__(self, offsets: list[tuple[int, int]]):
        self.offsets = offsets


class _WhitespaceTokenizer:
    """Small deterministic tokenizer fixture; no Hugging Face model is loaded."""

    def no_truncation(self) -> None:
        return None

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        return _Encoding([(match.start(), match.end()) for match in re.finditer(r"\S+", text)])


class _SemanticVectors:
    """Precomputed one-hot 384-dimensional vectors for retrieval semantics."""

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        # Axis 0 is the paraphrase pair: learned state / acquired information.
        if any(term in lowered for term in ("learned", "acquired", "external archive", "kept")):
            axis = 0
        # Axis 1 is deliberately unrelated to the paraphrase query.
        elif any(term in lowered for term in ("zqvulon", "calibration marker")):
            axis = 1
        else:
            axis = 2
        vector = [0.0] * 384
        vector[axis] = 1.0
        return vector

    def passages(self, texts):
        for text in texts:
            yield self._vector(text)

    def query(self, text: str) -> list[float]:
        # Exact-marker queries intentionally have no matching vector axis: the
        # marker regression must be satisfied by native FTS, not this fixture.
        lowered = text.lower()
        if "zqvulon" in lowered or "calibration marker" in lowered:
            axis = 3
            vector = [0.0] * 384
            vector[axis] = 1.0
            return vector
        return self._vector(text)

def _write_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page()
            page.insert_text((72, 72), text, fontsize=10)
        document.save(path)
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Settings, dict[str, Path]]:
    settings = Settings(data_dir=tmp_path / "data", pdf_dir=tmp_path / "papers")
    paths = {
        "archive": _write_pdf(
            settings.pdf_dir / "archive.pdf",
            ["A system preserves learned state in a compact external archive."],
        ),
        "marker": _write_pdf(
            settings.pdf_dir / "marker.pdf",
            [
                "Introductory notes contain no calibration finding.",
                "The calibration record includes the zqvulon marker.",
            ],
        ),
    }
    ingestor = Ingestor(settings, embedder=_SemanticVectors(), tokenizer=_WhitespaceTokenizer())
    report = ingestor.ingest_paths(paths.values())
    assert not report.failed, report
    return settings, paths


def _doc(path: Path) -> str:
    return document_id(path)


def test_native_fts_finds_rare_exact_term_and_vector_finds_paraphrase(corpus):
    settings, paths = corpus
    with RetrievalSession(settings, embedder=_SemanticVectors()) as session:
        exact = session.retrieve("zqvulon", top_k=2)
        assert exact
        assert exact[0].source_path == str(paths["marker"].resolve())
        assert "zqvulon" in exact[0].text.lower()

        paraphrase = session.retrieve(
            "Where is acquired information kept when the system cannot rely on itself?",
            top_k=2,
        )
        assert paraphrase
        assert paraphrase[0].source_path == str(paths["archive"].resolve())
        assert "external archive" in paraphrase[0].text.lower()


def test_document_prefilter_applies_to_hybrid_candidates(corpus):
    settings, paths = corpus
    archive_id = _doc(paths["archive"])
    marker_id = _doc(paths["marker"])
    with RetrievalSession(settings, embedder=_SemanticVectors(), document_ids=[marker_id]) as marker_scope:
        results = marker_scope.retrieve(
            "Where is acquired information kept when the system cannot rely on itself?", top_k=10
        )
        assert results
        assert {item.document_id for item in results} == {marker_id}
        assert all(item.source_path == str(paths["marker"].resolve()) for item in results)

    with RetrievalSession(settings, embedder=_SemanticVectors(), document_ids=[archive_id]) as archive_scope:
        results = archive_scope.retrieve("Which record names the zqvulon calibration marker?", top_k=10)
        assert results
        assert {item.document_id for item in results} == {archive_id}
        assert all(item.source_path == str(paths["archive"].resolve()) for item in results)


def test_session_version_is_pinned_but_fresh_session_sees_added_and_replaced_papers(corpus):
    settings, paths = corpus
    with RetrievalSession(settings, embedder=_SemanticVectors()) as pinned:
        before = pinned.retrieve("What is the zqvulon calibration marker?", top_k=5)
        assert before and before[0].source_path == str(paths["marker"].resolve())

        added = _write_pdf(
            settings.pdf_dir / "added.pdf",
            ["A newly indexed note describes acquired results in an external archive."],
        )
        report = Ingestor(settings, embedder=_SemanticVectors(), tokenizer=_WhitespaceTokenizer()).ingest_paths([added])
        assert not report.failed, report
        old_view = pinned.retrieve("Where are acquired results kept?", top_k=10)
        assert str(added.resolve()) not in {item.source_path for item in old_view}

        added_id = _doc(added)
        with RetrievalSession(settings, embedder=_SemanticVectors()) as fresh:
            new_view = fresh.retrieve("Where are acquired results kept?", top_k=10)
            added_results = [item for item in new_view if item.source_path == str(added.resolve())]
            assert added_results
            assert added_results[0].document_id == added_id

    # Replacing the bytes at an existing path must expose only the fresh revision
    # to a newly opened reader.
    replacement = _write_pdf(
        settings.pdf_dir / "marker-replacement.pdf",
        ["The zqvulon marker was replaced by a newly observed calibration signal."],
    )
    marker_path = paths["marker"]
    replacement.replace(marker_path)
    report = Ingestor(settings, embedder=_SemanticVectors(), tokenizer=_WhitespaceTokenizer()).ingest_paths([marker_path])
    assert not report.failed, report
    with RetrievalSession(settings, embedder=_SemanticVectors()) as fresh:
        results = fresh.retrieve("What is the newly observed zqvulon calibration signal?", top_k=10)
        replacement_results = [item for item in results if item.source_path == str(marker_path.resolve())]
        assert replacement_results
        assert any("newly observed" in item.text.lower() for item in replacement_results)



def _evidence(chunk_id: str, document: str, start: int, end: int, rank: int = 1) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        document_id=document,
        content_sha256="f" * 64,
        source_path=f"/tmp/{document}.pdf",
        title="fixture",
        page_number=1,
        chunk_ordinal=rank - 1,
        char_start=start,
        char_end=end,
        extraction_method="native",
        text=f"chunk {chunk_id}",
        rank=rank,
        score=1.0,
    )


def test_overlap_suppression_diversity_and_deterministic_fusion_ties():
    first_document = "1" * 64
    second_document = "2" * 64
    candidates = [
        _evidence("b" * 64, first_document, 0, 100),
        _evidence("c" * 64, first_document, 25, 75),  # > half-overlap with the first chunk
        _evidence("d" * 64, first_document, 100, 180),
        _evidence("e" * 64, first_document, 180, 260),
        _evidence("f" * 64, first_document, 260, 340),
        _evidence("g" * 64, first_document, 340, 420),  # fifth paper chunk is capped
        _evidence("a" * 64, second_document, 0, 50),
    ]
    selected = diverse(candidates, top_k=10)
    selected_ids = [item.chunk_id for item in selected]
    assert selected_ids == ["b" * 64, "d" * 64, "e" * 64, "f" * 64, "a" * 64]
    assert [item.rank for item in selected] == list(range(1, len(selected) + 1))

    tie_left = _evidence("b" * 64, first_document, 0, 10)
    tie_right = _evidence("a" * 64, second_document, 0, 10)
    fused = fuse([[tie_left], [tie_right]], top_k=2)
    assert [item.chunk_id for item in fused] == ["a" * 64, "b" * 64]
    assert [item.rank for item in fused] == [1, 2]


def test_retrieval_validates_question_top_k_and_document_ids(corpus):
    settings, paths = corpus
    with RetrievalSession(settings, embedder=_SemanticVectors()) as session:
        with pytest.raises(ValueError, match="blank"):
            session.retrieve("   ")
        with pytest.raises(ValueError, match="between 1 and 50"):
            session.retrieve("marker", top_k=0)
        with pytest.raises(ValueError, match="between 1 and 50"):
            session.retrieve("marker", top_k=51)
        with pytest.raises(ValueError, match="Set document_ids"):
            session.retrieve("marker", document_ids=[document_id(paths["marker"])])

    with pytest.raises(ValueError, match="Unknown document ID"):
        RetrievalSession(settings, embedder=_SemanticVectors(), document_ids=["0" * 64])
    with pytest.raises(ValueError, match="Unknown document ID"):
        RetrievalSession(settings, embedder=_SemanticVectors(), document_ids=["not-a-document-id"])


def test_source_provenance_uses_indexed_chunk_page_and_hash_and_rejects_changes(corpus):
    settings, paths = corpus
    with RetrievalSession(settings, embedder=_SemanticVectors()) as session:
        evidence = session.retrieve("zqvulon", top_k=1)[0]
    assert evidence.source_path == str(paths["marker"].resolve())
    assert evidence.page_number == 2
    assert evidence.char_end > evidence.char_start
    assert evidence.content_sha256 == file_hash(paths["marker"])
    assert source_path(settings, evidence) == paths["marker"].resolve()
    forged = evidence.model_copy(update={"page_number": 1})
    with pytest.raises(ValueError, match="Source does not match indexed evidence"):
        source_path(settings, forged)
    rendered_path, png = render_page(settings, evidence)
    assert rendered_path == paths["marker"].resolve()
    assert png.startswith(b"\x89PNG")

    paths["marker"].write_bytes(b"not the indexed PDF")
    with pytest.raises(ValueError, match=re.escape(UNAVAILABLE)):
        source_path(settings, evidence)
    paths["marker"].unlink()
    with pytest.raises(ValueError, match=re.escape(UNAVAILABLE)):
        render_page(settings, evidence)
