"""
Shared fastembed model (ONNX, no PyTorch) — used for both /ingest/embed's
product vectors (vector_store.py) and the semantic NL->SQL cache (llm.py), so
the model loads once per process instead of twice.
"""
from __future__ import annotations

from fastembed import TextEmbedding

from core.config import settings

_model: TextEmbedding | None = None


def get_embed_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(settings.embed_model)
    return _model


def embed_one(text: str) -> list[float]:
    return next(iter(get_embed_model().embed([text]))).tolist()


def embed_many(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in get_embed_model().embed(texts)]
