"""Lance is authoritative evidence; SQLite journals the whole-paper commit boundary."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import lancedb
import pyarrow as pa
from filelock import FileLock, Timeout

from .config import DIMENSION, Settings

SCHEMA_VERSION = 1
SCHEMA = pa.schema([
    *[(name, pa.string()) for name in ("chunk_id", "document_id", "content_sha256", "source_path", "title")],
    ("page_number", pa.int32()), ("chunk_ordinal", pa.int32()),
    ("char_start", pa.int64()), ("char_end", pa.int64()),
    ("extraction_method", pa.string()), ("text", pa.string()),
    ("pipeline_fingerprint", pa.string()), ("vector", pa.list_(pa.float32(), DIMENSION)),
])
MAX_PARTITIONS = 256


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def document_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.sql = sqlite3.connect(settings.data_dir / "manifest.sqlite3", timeout=10)
        self.sql.row_factory = sqlite3.Row
        self.sql.execute("PRAGMA journal_mode=WAL")
        self.sql.execute("PRAGMA foreign_keys=ON")
        self.sql.execute("PRAGMA busy_timeout=10000")
        self.sql.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY, source_path TEXT NOT NULL UNIQUE,
                active_json TEXT, pending_json TEXT);
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, error TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)
        self.db = lancedb.connect(str(settings.data_dir / "lancedb"), read_consistency_interval=timedelta(0))
        self.identity = json.dumps({"schema": SCHEMA_VERSION, "model": settings.embedding_model, "dimension": DIMENSION}, sort_keys=True)
        row = self.sql.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row and row[0] != self.identity:
            self.close()
            raise ValueError("Index embedding identity/schema is incompatible; choose a new --data-dir. Existing evidence was not modified.")
        self.table = self.db.open_table("chunks") if "chunks" in self.db.list_tables().tables else None
        if self.table is not None:
            if not self.table.schema.equals(SCHEMA, check_metadata=False):
                self.close()
                raise ValueError("Incompatible chunks schema; choose a new --data-dir")
            if not row and self.table.count_rows():
                self.close()
                raise ValueError("Nonempty index has no embedding identity; choose a new --data-dir")

    def close(self) -> None:
        self.sql.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @contextmanager
    def writer(self) -> Iterator[None]:
        lock = FileLock(str(self.settings.data_dir / "write.lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise ValueError("Index is busy; another writer is running.") from exc
        try:
            with self.sql:
                self.sql.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (self.identity,))
            if self.table is None:
                self.table = self.db.create_table("chunks", schema=SCHEMA, exist_ok=True)
            self.reconcile()
            yield
        finally:
            lock.release()

    def active(self, doc_id: str) -> dict | None:
        row = self.sql.execute("SELECT active_json FROM documents WHERE document_id=?", (doc_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def duplicate(self, digest: str) -> str | None:
        row = self.sql.execute("""SELECT document_id FROM documents WHERE active_json IS NOT NULL
            AND json_extract(active_json, '$.content_sha256')=?
            ORDER BY source_path LIMIT 1""", (digest,)).fetchone()
        return row[0] if row else None

    def record_failure(self, path: str, error: str) -> None:
        with self.sql:
            self.sql.execute("INSERT INTO failures(source_path,error) VALUES (?,?)", (path, error))

    def begin_pending(self, meta: dict) -> None:
        with self.sql:
            self.sql.execute("""INSERT INTO documents(document_id,source_path,pending_json) VALUES (?,?,?)
                ON CONFLICT(document_id) DO UPDATE SET pending_json=excluded.pending_json""",
                (meta["document_id"], meta["source_path"], json.dumps(meta)))

    def finish_pending(self, doc_id: str, meta: dict) -> None:
        with self.sql:
            self.sql.execute("UPDATE documents SET active_json=?,pending_json=NULL WHERE document_id=?", (json.dumps(meta), doc_id))

    def reconcile(self) -> None:
        """Never guess whether Lance committed from an exception in another database."""
        if self.table is None:
            return
        self.table.checkout_latest()
        pending = self.sql.execute("SELECT document_id,source_path,pending_json FROM documents WHERE pending_json IS NOT NULL").fetchall()
        for row in pending:
            meta = json.loads(row["pending_json"])
            doc_id = row["document_id"]
            where = f"document_id = '{doc_id}'"
            count = self.table.count_rows(where)
            matching = self.table.count_rows(where + f" AND content_sha256 = '{meta['content_sha256']}' AND pipeline_fingerprint = '{meta['pipeline_fingerprint']}'")
            if count == matching == meta["chunks"] and count > 0:
                self.finish_pending(doc_id, meta)
            else:
                with self.sql:
                    self.sql.execute("UPDATE documents SET pending_json=NULL WHERE document_id=?", (doc_id,))
                    self.sql.execute("DELETE FROM documents WHERE document_id=? AND active_json IS NULL", (doc_id,))
                self.record_failure(row["source_path"], "Interrupted indexing attempt; prior committed revision preserved")

    def commit_staged(self, meta: dict) -> None:
        assert self.table is not None
        self.begin_pending(meta)
        with pa.memory_map(meta["stage_path"], "r") as source:
            reader = pa.ipc.open_stream(source)
            (self.table.merge_insert("chunk_id").when_matched_update_all()
             .when_not_matched_insert_all()
             .when_not_matched_by_source_delete(f"document_id = '{meta['document_id']}'")
             .execute(reader))
        self.finish_pending(meta["document_id"], meta)

    def stats(self) -> dict:
        row = self.sql.execute("""SELECT count(*),coalesce(sum(json_extract(active_json,'$.pages')),0),
            coalesce(sum(json_extract(active_json,'$.chunks')),0),coalesce(sum(json_extract(active_json,'$.empty_pages')),0)
            FROM documents WHERE active_json IS NOT NULL""").fetchone()
        indexes = []
        if self.table is not None:
            self.table.checkout_latest()
            for index in self.table.list_indices():
                stat = self.table.index_stats(index.name)
                indexes.append({"name": index.name, "type": str(index.index_type),
                                "indexed_rows": stat.num_indexed_rows if stat else 0,
                                "unindexed_rows": stat.num_unindexed_rows if stat else 0})
        return {"documents": row[0], "pages": row[1], "chunks": row[2], "empty_pages": row[3],
                "failures": self.sql.execute("SELECT count(*) FROM failures").fetchone()[0],
                "pending": self.sql.execute("SELECT count(*) FROM documents WHERE pending_json IS NOT NULL").fetchone()[0],
                "indexes": indexes, "data_dir": str(self.settings.data_dir)}

    def documents(self, *, offset: int = 0, limit: int = 25) -> list[dict]:
        rows = self.sql.execute("SELECT active_json FROM documents WHERE active_json IS NOT NULL ORDER BY source_path LIMIT ? OFFSET ?", (limit, offset))
        return [json.loads(row[0]) for row in rows]

    def maintain_indexes(self) -> dict:
        assert self.table is not None
        self.table.checkout_latest()
        count = self.table.count_rows()
        if not count:
            return self.stats()
        indexes = {tuple(i.columns): i for i in self.table.list_indices()}
        if ("text",) not in indexes:
            self.table.create_fts_index("text", use_tantivy=False)
        for column in ("chunk_id", "document_id"):
            if (column,) not in indexes:
                self.table.create_scalar_index(column)
        if count >= 10_000 and ("vector",) not in indexes:
            self.table.create_index(metric="cosine", index_type="IVF_PQ", num_sub_vectors=24,
                                    num_partitions=max(1, min(MAX_PARTITIONS, count // 256)), sample_rate=64)
        self.table.optimize()
        names = [i.name for i in self.table.list_indices()]
        self.table.wait_for_index(names, timeout=timedelta(seconds=600))
        for name in names:
            stat = self.table.index_stats(name)
            if stat is None or stat.num_unindexed_rows:
                raise RuntimeError(f"Index maintenance incomplete for {name}: unindexed rows remain")
        return self.stats()
