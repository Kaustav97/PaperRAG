from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pymupdf
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from paperrag.config import Settings
from paperrag.ingest import Ingestor
from paperrag.storage import Store, document_id, file_hash


class TinyEmbedder:
    """Deterministic 384-dimensional vectors without loading a model."""

    def passages(self, texts):
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            yield (list(digest) * 12)[:384]


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.no_truncation()
    return tokenizer


def _settings(tmp_path: Path, *, embedding_model: str = "tiny-local") -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "papers",
        embedding_model=embedding_model,
        embed_batch_size=16,
        threads=1,
    )


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    try:
        for lines in pages:
            page = document.new_page(width=612, height=792)
            for line_number, line in enumerate(lines):
                # Separate text blocks keep native extraction deterministic while
                # leaving enough room for a >384-token page.
                page.insert_text((36, 38 + line_number * 10), line, fontsize=7)
        document.save(path)
    finally:
        document.close()


def _ingest(settings: Settings, *paths: Path):
    return Ingestor(settings, embedder=TinyEmbedder(), tokenizer=_tokenizer()).ingest_paths(paths)


def _rows(settings: Settings) -> list[dict]:
    with Store(settings) as store:
        assert store.table is not None
        store.table.checkout_latest()
        return store.table.to_arrow().to_pylist()


def _rows_for(settings: Settings, doc_id: str) -> list[dict]:
    return [row for row in _rows(settings) if row["document_id"] == doc_id]


def _stage(settings: Settings, path: Path) -> dict:
    ingestor = Ingestor(settings, embedder=TinyEmbedder(), tokenizer=_tokenizer())
    ingestor._force_ocr = False
    digest = file_hash(path)
    return ingestor.stage(path, digest, ingestor.fingerprint(False))

def _stats(settings: Settings) -> dict:
    with Store(settings) as store:
        return store.stats()



def _assert_complete_revision(settings: Settings, doc_id: str, digest: str) -> None:
    rows = _rows_for(settings, doc_id)
    assert rows
    assert {row["content_sha256"] for row in rows} == {digest}
    assert len({row["chunk_id"] for row in rows}) == len(rows)
    with Store(settings) as store:
        active = store.active(doc_id)
    assert active is not None
    assert active["content_sha256"] == digest
    assert active["chunks"] == len(rows)


def test_pages_offsets_and_final_fragment_are_persisted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = tmp_path / "papers" / "boundary.pdf"
    page_one = [
        f"page one long passage line {number} alpha beta gamma delta epsilon"
        for number in range(68)
    ]
    page_one[-1] += " PAGEONEFINAL"
    page_two = [f"second page distinct phrase line {number}" for number in range(4)]
    page_two[-1] += " PAGETWOUNIQUE"
    _write_pdf(path, [page_one, page_two])

    report = _ingest(settings, path)
    assert [file.status for file in report.files] == ["indexed"]
    doc_id = document_id(path)
    rows = _rows_for(settings, doc_id)
    assert {row["page_number"] for row in rows} == {1, 2}
    page_one_rows = [row for row in rows if row["page_number"] == 1]
    page_two_rows = [row for row in rows if row["page_number"] == 2]
    assert len(page_one_rows) >= 2
    assert any("PAGEONEFINAL" in row["text"] for row in page_one_rows)
    assert any("PAGETWOUNIQUE" in row["text"] for row in page_two_rows)

    from paperrag.parsing import extract_pages

    pages = list(extract_pages(path, ocr_language="eng"))
    assert len(pages) == 2
    assert len(_tokenizer().encode(pages[0].text, add_special_tokens=False).ids) > 384
    page_one_rows = sorted(page_one_rows, key=lambda row: row["chunk_ordinal"])
    assert "PAGEONEFINAL" in page_one_rows[-1]["text"]
    for row in rows:
        page_text = pages[row["page_number"] - 1].text
        assert row["char_end"] <= len(page_text)
        assert row["text"] == page_text[row["char_start"] : row["char_end"]]


def test_shorter_replacement_removes_old_chunks_and_keeps_unrelated_document(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = tmp_path / "papers" / "first.pdf"
    unrelated = tmp_path / "papers" / "unrelated.pdf"
    _write_pdf(first, [["OLDONLY evidence that must be removed", "old context"]])
    _write_pdf(unrelated, [["UNRELATED evidence remains searchable"]])
    initial = _ingest(settings, first, unrelated)
    assert {result.status for result in initial.files} == {"indexed"}
    unrelated_id = document_id(unrelated)
    unrelated_before = {row["chunk_id"] for row in _rows_for(settings, unrelated_id)}

    _write_pdf(first, [["NEWONLY replacement evidence", "new context"]])
    replaced = _ingest(settings, first)
    assert replaced.files[0].status == "indexed"
    rows = _rows(settings)
    first_rows = [row for row in rows if row["document_id"] == document_id(first)]
    assert first_rows
    assert all("OLDONLY" not in row["text"] for row in first_rows)
    assert any("NEWONLY" in row["text"] for row in first_rows)
    assert {row["chunk_id"] for row in _rows_for(settings, unrelated_id)} == unrelated_before
    assert len({row["chunk_id"] for row in rows}) == len(rows)


@pytest.mark.parametrize("phase", ["before_merge", "after_commit"])
def test_pending_revision_recovery_and_retry_never_mixes_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    settings = _settings(tmp_path)
    path = tmp_path / "papers" / "recover.pdf"
    _write_pdf(path, [["OLDREVISION only text"]])
    _ingest(settings, path)
    old_digest = file_hash(path)
    doc_id = document_id(path)

    _write_pdf(path, [["NEWREVISION only replacement text"]])
    new_digest = file_hash(path)
    meta = _stage(settings, path)

    with Store(settings) as store:
        with store.writer():
            if phase == "before_merge":
                def fail_merge(_self, *args, **kwargs):
                    raise RuntimeError("injected interruption before Lance merge")

                monkeypatch.setattr(type(store.table), "merge_insert", fail_merge)
                with pytest.raises(RuntimeError, match="before Lance merge"):
                    store.commit_staged(meta)
            else:
                def fail_finish(_doc_id, _meta):
                    raise RuntimeError("injected interruption after Lance commit")

                monkeypatch.setattr(store, "finish_pending", fail_finish)
                with pytest.raises(RuntimeError, match="after Lance commit"):
                    store.commit_staged(meta)

    monkeypatch.undo()
    # A fresh writer is the recovery boundary: either the old committed revision
    # survives a pre-merge interruption, or the complete new Lance revision is
    # promoted when SQLite was interrupted after Lance committed.
    with Store(settings) as recovered:
        with recovered.writer():
            pass
    with Store(settings) as observed:
        active_after_recovery = observed.active(doc_id)
    assert active_after_recovery is not None
    assert active_after_recovery["content_sha256"] in {old_digest, new_digest}
    _assert_complete_revision(settings, doc_id, active_after_recovery["content_sha256"])

    retry = _ingest(settings, path)
    assert retry.files[0].status in {"indexed", "unchanged"}
    _assert_complete_revision(settings, doc_id, new_digest)
    final_rows = _rows_for(settings, doc_id)
    assert any("NEWREVISION" in row["text"] for row in final_rows)
    assert all("OLDREVISION" not in row["text"] for row in final_rows)


def test_unchanged_and_byte_duplicate_do_not_add_documents_or_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    original = tmp_path / "papers" / "original.pdf"
    duplicate = tmp_path / "papers" / "duplicate.pdf"
    _write_pdf(original, [["DUPLICATE BYTE CONTENT"]])
    first = _ingest(settings, original)
    assert first.files[0].status == "indexed"
    before = _stats(settings)
    shutil.copyfile(original, duplicate)

    second = _ingest(settings, original)
    third = _ingest(settings, duplicate)
    assert second.files[0].status == "unchanged"
    assert third.files[0].status == "duplicate"
    assert third.files[0].duplicate_of == document_id(original)
    after = _stats(settings)
    assert (after["documents"], after["chunks"]) == (before["documents"], before["chunks"]) == (1, first.files[0].chunks)


def test_corrupt_pdf_does_not_prevent_valid_file_and_is_reported(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    valid = tmp_path / "papers" / "valid.pdf"
    corrupt = tmp_path / "papers" / "corrupt.PDF"
    _write_pdf(valid, [["VALID CONTENT"]])
    corrupt.write_bytes(b"not a PDF")

    report = _ingest(settings, tmp_path / "papers")
    results = {Path(result.source_path).name: result for result in report.files}
    assert results[valid.name].status == "indexed"
    assert results[corrupt.name].status == "failed"
    assert results[corrupt.name].error
    with Store(settings) as store:
        stats = store.stats()
    assert stats["documents"] == 1
    assert stats["failures"] >= 1


def test_changed_path_replaces_even_when_new_bytes_duplicate_another_document(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = tmp_path / "papers" / "first.pdf"
    second = tmp_path / "papers" / "second.pdf"
    _write_pdf(first, [["FIRST OLD CONTENT"]])
    _write_pdf(second, [["SECOND CONTENT THAT BECOMES FIRST"]])
    _ingest(settings, first, second)

    shutil.copyfile(second, first)
    report = _ingest(settings, first)
    assert report.files[0].status == "indexed"
    first_rows = _rows_for(settings, document_id(first))
    second_rows = _rows_for(settings, document_id(second))
    assert first_rows and second_rows
    assert any("SECOND CONTENT" in row["text"] for row in first_rows)
    assert all("FIRST OLD" not in row["text"] for row in first_rows)
    assert len({row["document_id"] for row in _rows(settings)}) == 2
    assert len({row["chunk_id"] for row in _rows(settings)}) == len(_rows(settings))


def test_nonmatching_embedding_identity_is_refused(tmp_path: Path) -> None:
    settings = _settings(tmp_path, embedding_model="tiny-model-a")
    path = tmp_path / "papers" / "identity.pdf"
    _write_pdf(path, [["IDENTITY CONTENT"]])
    _ingest(settings, path)

    incompatible = _settings(tmp_path, embedding_model="tiny-model-b")
    with pytest.raises(ValueError, match="incompatible.*data-dir"):
        Store(incompatible)
