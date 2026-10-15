"""Source paths are resolved exclusively from authoritative indexed chunk IDs."""
from __future__ import annotations
import re
from pathlib import Path
import pymupdf
from .models import Evidence
from .storage import Store, file_hash
from .config import Settings

UNAVAILABLE = "Original PDF is missing or changed; re-ingest to refresh."


def source_path(settings: Settings, evidence: Evidence) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", evidence.chunk_id):
        raise ValueError("Unknown indexed source")
    with Store(settings) as store:
        if store.table is None:
            raise ValueError("Unknown indexed source")
        rows = store.table.search().where(f"chunk_id = '{evidence.chunk_id}'").select(
            ["source_path", "content_sha256", "page_number", "document_id"]).limit(1).to_list()
    # A previous pinned result may have been replaced meanwhile; never show a new PDF for it.
    if not rows:
        raise ValueError(UNAVAILABLE)
    row = rows[0]
    if (row["document_id"], row["page_number"], row["content_sha256"]) != (evidence.document_id, evidence.page_number, evidence.content_sha256):
        raise ValueError("Source does not match indexed evidence")
    path = Path(row["source_path"])
    try:
        if file_hash(path) != row["content_sha256"]:
            raise ValueError(UNAVAILABLE)
    except OSError as exc:
        raise ValueError(UNAVAILABLE) from exc
    return path


def render_page(settings: Settings, evidence: Evidence) -> tuple[Path, bytes]:
    path = source_path(settings, evidence)
    with pymupdf.open(path) as document:
        if not 1 <= evidence.page_number <= len(document):
            raise ValueError(UNAVAILABLE)
        png = document[evidence.page_number - 1].get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2), alpha=False).tobytes("png")
    # Detect a source replacement while rendering as well.
    if file_hash(path) != evidence.content_sha256:
        raise ValueError(UNAVAILABLE)
    return path, png
