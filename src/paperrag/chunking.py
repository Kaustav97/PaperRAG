"""Token windows slice canonical page text, never decoded token IDs."""
from __future__ import annotations

from typing import Iterable, Iterator
from tokenizers import Tokenizer
from .config import Settings
from .models import PageText, Chunk

CHUNKER_VERSION = "page-offset-384-64-v1"
WINDOW = 384
OVERLAP = 64


def load_tokenizer(settings: Settings) -> Tokenizer:
    settings.prepare_models()
    tokenizer = Tokenizer.from_pretrained(settings.embedding_model)
    tokenizer.no_truncation()
    return tokenizer


def chunk_pages(pages: Iterable[PageText], tokenizer: Tokenizer) -> Iterator[Chunk]:
    tokenizer.no_truncation()
    for page in pages:
        if not page.text.strip():
            continue
        offsets = tokenizer.encode(page.text, add_special_tokens=False).offsets
        start, ordinal = 0, 0
        while start < len(offsets):
            end = min(start + WINDOW, len(offsets))
            if end < len(offsets):
                # Prefer the latest paragraph break in the last overlap-width tokens.
                for candidate in range(end, max(start + OVERLAP, end - OVERLAP), -1):
                    left, right = offsets[candidate - 1][1], offsets[candidate][0]
                    if "\n\n" in page.text[left:right]:
                        end = candidate
                        break
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            if char_end > char_start:
                yield Chunk(source_path=page.source_path, title=page.title, page_number=page.page_number,
                            chunk_ordinal=ordinal, char_start=char_start, char_end=char_end,
                            extraction_method=page.extraction_method, text=page.text[char_start:char_end])
                ordinal += 1
            if end == len(offsets):
                break
            start = max(start + 1, end - OVERLAP)
