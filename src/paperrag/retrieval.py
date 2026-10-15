"""Explicit-vector native hybrid retrieval, pinned per research run."""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from lancedb.rerankers import RRFReranker
from .config import Settings, load_settings
from .embedding import Embedder
from .models import Evidence
from .storage import MAX_PARTITIONS, SCHEMA, Store

EVIDENCE_FIELDS = [n for n in SCHEMA.names if n not in ("vector", "pipeline_fingerprint")]


def diverse(evidence: Sequence[Evidence], top_k: int) -> list[Evidence]:
    result, seen, counts = [], set(), Counter()
    for e in evidence:
        if e.chunk_id in seen or counts[e.document_id] >= 4:
            continue
        overlap = False
        for other in result:
            if (other.document_id, other.page_number) == (e.document_id, e.page_number):
                intersection = max(0, min(other.char_end, e.char_end) - max(other.char_start, e.char_start))
                if intersection > min(other.char_end - other.char_start, e.char_end - e.char_start) / 2:
                    overlap = True
                    break
        if overlap:
            continue
        seen.add(e.chunk_id)
        counts[e.document_id] += 1
        result.append(e.model_copy(update={"rank": len(result) + 1}))
        if len(result) == top_k:
            break
    return result


def fuse(rankings: Sequence[Sequence[Evidence]], top_k: int = 16) -> list[Evidence]:
    scores, items = Counter(), {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item.chunk_id not in seen:
                scores[item.chunk_id] += 1 / (60 + rank)
                items[item.chunk_id] = item
                seen.add(item.chunk_id)
    ordered = sorted(items, key=lambda key: (-scores[key], key))
    return diverse([items[key].model_copy(update={"score": scores[key]}) for key in ordered], top_k)


class RetrievalSession:
    def __init__(self, settings: Settings, *, embedder=None, document_ids: Sequence[str] | None = None):
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.store = Store(settings)
        self.table = None
        self.filter = None
        try:
            if document_ids is not None:
                for doc_id in document_ids:
                    if not re.fullmatch(r"[0-9a-f]{64}", doc_id) or not self.store.active(doc_id):
                        raise ValueError(f"Unknown document ID: {doc_id}")
                self.filter = "document_id IN (" + ",".join(f"'{d}'" for d in document_ids) + ")" if document_ids else "false"
            if self.store.table is not None:
                # A dedicated handle, refreshed once and then version-pinned for all rounds.
                self.table = self.store.db.open_table("chunks")
                self.table.checkout_latest()
                self.version = self.table.version
                self.table.checkout(self.version)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def retrieve(self, question: str, *, top_k: int = 12, document_ids: Sequence[str] | None = None) -> list[Evidence]:
        if not question.strip():
            raise ValueError("Question must not be blank")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if document_ids is not None:
            raise ValueError("Set document_ids when opening a RetrievalSession so expansion uses the same scope")
        if self.table is None or not self.table.count_rows() or self.filter == "false":
            return []
        query = (self.table.search(query_type="hybrid").vector(self.embedder.query(question)).text(question)
                 .rerank(RRFReranker()).metric("cosine").limit(min(200, 4 * top_k)).select(EVIDENCE_FIELDS))
        vector_index = next((i for i in self.table.list_indices() if "vector" in i.columns), None)
        if vector_index:
            # Scale smoke required all partitions and deeper exact refinement.
            # Lance clamps this build-time ceiling to the actual partition count.
            query = query.nprobes(MAX_PARTITIONS).refine_factor(20)
        if self.filter:
            query = query.where(self.filter, prefilter=True)
        rows = query.to_list()
        evidence = []
        for rank, row in enumerate(rows, 1):
            fields = {key: row[key] for key in EVIDENCE_FIELDS}
            evidence.append(Evidence(**fields, rank=rank, score=float(row["_relevance_score"])))
        return diverse(evidence, top_k)


def retrieve(question: str, *, top_k: int = 12, document_ids: Sequence[str] | None = None) -> list[Evidence]:
    with RetrievalSession(load_settings(), document_ids=document_ids) as session:
        return session.retrieve(question, top_k=top_k)
