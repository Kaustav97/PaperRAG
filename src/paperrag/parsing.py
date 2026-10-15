"""Bounded-memory, physical-page PDF extraction. Evidence stays in page coordinates."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import pymupdf

from .config import tessdata_dir
from .models import PageText

PARSER_VERSION = "native-blocks-ocr-v1"
SAMPLE_TITLES = {
    "1706.03762": "Attention Is All You Need",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2307.03172": "Lost in the Middle: How Language Models Use Long Contexts",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def known_title(path: Path) -> str | None:
    # Only the sample provenance maps arbitrary downloaded paths to known titles.
    manifest = path.parent / "samples.json"
    if manifest.is_file():
        try:
            for record in json.loads(manifest.read_text()):
                if path.name == record["paper_id"] + ".pdf":
                    return SAMPLE_TITLES.get(record["paper_id"])
        except (ValueError, KeyError, TypeError):
            pass
    return SAMPLE_TITLES.get(path.stem)


def _block_text(blocks: list) -> str:
    return normalize_text("\n\n".join(b[4] for b in blocks if len(b) > 6 and b[6] == 0 and b[4].strip()))


def extract_pages(path: Path, *, ocr_language: str, force_ocr: bool = False) -> Iterator[PageText]:
    path = path.resolve()
    with pymupdf.open(path) as doc:
        if doc.needs_pass or doc.is_encrypted:
            raise ValueError("PDF is encrypted/password-protected")
        metadata = {str(k): str(v) if v is not None else None for k, v in (doc.metadata or {}).items()}
        title = known_title(path) or (metadata.get("title") or "").strip() or path.stem
        for page in doc:
            text = _block_text(page.get_text("blocks", sort=True))
            method = "native"
            # Avoid building image payloads; inspect image placement rectangles only.
            image_rects = [pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]
            coverage = min(page.rect.get_area(), sum(r.get_area() for r in image_rects)) / max(1, page.rect.get_area())
            sparse = sum(c.isalnum() for c in text) < 40
            needs_ocr = force_ocr or (sparse and coverage >= 0.5) or (not text and (image_rects or page.get_drawings()))
            if needs_ocr:
                directory = tessdata_dir()
                missing = [lang for lang in ocr_language.split("+") if not (directory / (lang + ".traineddata")).is_file()]
                if missing:
                    raise ValueError(f"OCR required on page {page.number + 1}; missing traineddata {missing} in {directory}")
                tp = page.get_textpage_ocr(language=ocr_language, dpi=200, full=True, tessdata=str(directory))
                text = _block_text(page.get_text("blocks", textpage=tp, sort=True))
                method = "ocr"
            yield PageText(source_path=str(path), title=title, page_number=page.number + 1,
                           text=text, extraction_method=method, metadata=metadata)
