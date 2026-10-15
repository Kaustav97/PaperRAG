"""Lazy CPU ONNX embedding with strict vector validation."""
from __future__ import annotations
from typing import Iterable, Iterator
import numpy as np
from .config import DIMENSION, Settings


def valid_vector(value) -> list[float]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (DIMENSION,) or not np.isfinite(array).all() or not np.any(array):
        raise ValueError(f"Embedding must be a finite, nonzero {DIMENSION}-vector")
    return array.tolist()


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self.settings.prepare_models()
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.settings.embedding_model,
                                        threads=self.settings.threads,
                                        cache_dir=str(self.settings.data_dir / "models" / "fastembed"), cuda=False)
        return self._model

    def passages(self, texts: Iterable[str]) -> Iterator[list[float]]:
        for vector in self.model.passage_embed(texts, batch_size=self.settings.embed_batch_size, parallel=None):
            yield valid_vector(vector)

    def query(self, text: str) -> list[float]:
        return valid_vector(next(iter(self.model.query_embed(text))))
