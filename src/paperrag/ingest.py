"""One PDF, page and bounded embedding batch at a time; originals are read-only."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pyarrow as pa
from .chunking import CHUNKER_VERSION, chunk_pages, load_tokenizer
from .config import DIMENSION, Settings, load_settings
from .embedding import Embedder, valid_vector
from .models import FileResult, IngestReport, StageEvent
from .parsing import PARSER_VERSION, extract_pages
from .storage import SCHEMA, Store, document_id, file_hash


def discover(paths: Iterable[Path]) -> Iterator[Path]:
    found = set()
    for given in paths:
        path = given.expanduser()
        if not path.exists():
            found.add(path.resolve())
        elif path.is_dir():
            for root, directories, files in os.walk(path, followlinks=False):
                directories[:] = sorted(d for d in directories if not (Path(root) / d).is_symlink())
                for name in files:
                    if Path(name).suffix.lower() == ".pdf":
                        found.add((Path(root) / name).resolve())
        else:
            found.add(path.resolve())
    yield from sorted(found, key=str)


class Ingestor:
    def __init__(self, settings: Settings, *, embedder=None, tokenizer=None,
                 on_progress: Callable[[StageEvent], None] | None = None):
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.tokenizer = tokenizer
        self.on_progress = on_progress

    def emit(self, message: str) -> None:
        if self.on_progress:
            self.on_progress(StageEvent(stage="index", message=message))

    def fingerprint(self, force_ocr: bool) -> str:
        data = [PARSER_VERSION, CHUNKER_VERSION, self.settings.ocr_language,
                force_ocr, self.settings.embedding_model, DIMENSION]
        return hashlib.sha256(json.dumps(data).encode()).hexdigest()

    def stage(self, path: Path, digest: str, fingerprint: str) -> dict:
        doc_id = document_id(path)
        directory = self.settings.data_dir / "cache" / doc_id / (digest + "-" + fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        pages_path, arrow_path = directory / "pages.jsonl", directory / "chunks.arrow"
        pages_tmp, arrow_tmp = directory / "pages.tmp", directory / "chunks.tmp"
        meta = dict(document_id=doc_id, source_path=str(path), content_sha256=digest,
                    pipeline_fingerprint=fingerprint, pages=0, empty_pages=0, chunks=0, title=path.stem,
                    stage_path=str(arrow_path), pages_path=str(pages_path))
        try:
            with pages_tmp.open("w") as page_file, pa.OSFile(str(arrow_tmp), "wb") as sink:
                def pages():
                    for page in extract_pages(path, ocr_language=self.settings.ocr_language, force_ocr=self._force_ocr):
                        meta["pages"] += 1
                        meta["empty_pages"] += int(not page.text.strip())
                        meta["title"] = page.title
                        page_file.write(page.model_dump_json() + "\n")
                        self.emit(f"{path.name}: page {page.page_number} ({page.extraction_method})")
                        yield page
                if self.tokenizer is None:
                    self.tokenizer = load_tokenizer(self.settings)
                with pa.ipc.new_stream(sink, SCHEMA) as writer:
                    batch = []
                    def flush():
                        vectors = iter(self.embedder.passages(c.text for c in batch))
                        rows = []
                        for c in batch:
                            identity = [doc_id, digest, fingerprint, c.page_number, c.chunk_ordinal]
                            rows.append({**c.model_dump(), "chunk_id": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
                                         "document_id": doc_id, "content_sha256": digest, "pipeline_fingerprint": fingerprint,
                                         "vector": valid_vector(next(vectors))})
                        if next(vectors, None) is not None:
                            raise ValueError("Embedding provider returned excess vectors")
                        writer.write_batch(pa.RecordBatch.from_pylist(rows, schema=SCHEMA))
                        meta["chunks"] += len(rows)
                        batch.clear()
                    for chunk in chunk_pages(pages(), self.tokenizer):
                        batch.append(chunk)
                        if len(batch) == self.settings.embed_batch_size:
                            flush()
                    if batch:
                        flush()
            if not meta["chunks"]:
                raise ValueError("PDF is entirely empty; no searchable text extracted")
            if file_hash(path) != digest:
                raise ValueError("PDF changed during indexing; retry when the source is stable")
            pages_tmp.replace(pages_path)
            arrow_tmp.replace(arrow_path)
            return meta
        except BaseException:
            pages_tmp.unlink(missing_ok=True)
            arrow_tmp.unlink(missing_ok=True)
            raise

    def ingest_paths(self, paths: Iterable[Path], *, force_ocr: bool = False) -> IngestReport:
        started = time.monotonic()
        report = IngestReport()
        self._force_ocr = force_ocr
        fingerprint = self.fingerprint(force_ocr)
        with Store(self.settings) as store, store.writer():
            commits = 0
            for path in discover(paths):
                doc_id = document_id(path)
                self.emit(f"Indexing {path}")
                try:
                    digest = file_hash(path)
                    active = store.active(doc_id)
                    if active and active["content_sha256"] == digest and active["pipeline_fingerprint"] == fingerprint:
                        report.files.append(FileResult(source_path=str(path), status="unchanged", document_id=doc_id,
                                                       pages=active["pages"], chunks=active["chunks"]))
                        continue
                    duplicate = None if active else store.duplicate(digest)
                    if duplicate:
                        report.files.append(FileResult(source_path=str(path), status="duplicate", document_id=doc_id, duplicate_of=duplicate))
                        continue
                    meta = self.stage(path, digest, fingerprint)
                    store.commit_staged(meta)
                    report.files.append(FileResult(source_path=str(path), status="indexed", document_id=doc_id,
                                                   pages=meta["pages"], chunks=meta["chunks"]))
                    commits += 1
                    if commits % 20 == 0:
                        try:
                            store.maintain_indexes()
                        except Exception as exc:
                            report.maintenance_error = f"Index maintenance failed: {exc}"
                except Exception as exc:
                    # This may finish a successful Lance commit, even after SQLite failed.
                    store.reconcile()
                    error = f"{type(exc).__name__}: {exc}"
                    store.record_failure(str(path), error)
                    report.files.append(FileResult(source_path=str(path), status="failed", document_id=doc_id, error=error))
            try:
                store.maintain_indexes()
            except Exception as exc:
                report.maintenance_error = f"Index maintenance failed: {exc}"
        report.elapsed_seconds = round(time.monotonic() - started, 3)
        return report


def ingest_paths(paths: Iterable[Path], *, force_ocr: bool = False) -> IngestReport:
    return Ingestor(load_settings()).ingest_paths(paths, force_ocr=force_ocr)
