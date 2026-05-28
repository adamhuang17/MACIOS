"""Embedding provider abstraction for RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

EmbeddingProvider = Literal["local", "openai", "openai_compatible"]


class Embedder:
    """Text embedding service.

    ``provider="local"`` uses sentence-transformers. ``provider="openai"`` and
    ``provider="openai_compatible"`` use the OpenAI SDK against a configurable
    embeddings endpoint.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str | None = None,
        *,
        provider: str = "local",
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._provider = self._normalize_provider(provider)
        self._api_key = api_key
        self._base_url = base_url
        self._dimension = dimension
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._timeout = timeout
        self._model: SentenceTransformer | None = None
        self._client: OpenAI | None = None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return configured or discovered embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        if self._provider == "local":
            self._ensure_local_model()
            assert self._dimension is not None  # noqa: S101
            return self._dimension
        raise ValueError(
            "embedding_dimension must be configured for remote embedding providers"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        if self._provider == "local":
            return self._embed_local(texts)
        return self._embed_remote(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed([text])[0]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_local_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        result = embeddings.tolist()  # type: ignore[union-attr]
        if result and self._dimension is None:
            self._dimension = len(result[0])
        return result

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_remote_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = client.embeddings.create(model=self._model_name, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            for item in ordered:
                vector = [float(value) for value in item.embedding]
                vectors.append(self._maybe_normalize(vector))
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def _ensure_local_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        device = self._device or self._detect_device()
        logger.info("embedder_loading", provider="local", model=self._model_name, device=device)
        self._model = SentenceTransformer(self._model_name, device=device)
        dimension = self._model.get_sentence_embedding_dimension()
        self._dimension = dimension  # type: ignore[assignment]
        logger.info(
            "embedder_loaded",
            provider="local",
            model=self._model_name,
            dimension=self._dimension,
        )
        return self._model

    def _ensure_remote_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        from openai import OpenAI

        kwargs: dict[str, str] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(timeout=self._timeout, **kwargs)
        logger.info(
            "embedder_loaded",
            provider=self._provider,
            model=self._model_name,
            base_url=self._base_url,
            dimension=self._dimension,
        )
        return self._client

    def _maybe_normalize(self, vector: list[float]) -> list[float]:
        if not self._normalize:
            return vector
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        value = (provider or "local").strip().lower().replace("-", "_")
        if value in {"local", "sentence_transformers", "sentence_transformer"}:
            return "local"
        if value in {"openai", "openai_compatible", "remote"}:
            return "openai_compatible" if value == "remote" else value
        raise ValueError(f"Unsupported embedding provider: {provider}")

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
