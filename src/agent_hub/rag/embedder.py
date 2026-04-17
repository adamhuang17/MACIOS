"""Embedding 服务封装。

支持 BGE-large-zh / text2vec-large-chinese 等 sentence-transformers 模型，
本地推理，设备自动检测（CUDA / CPU）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class Embedder:
    """文本向量化服务，基于 sentence-transformers。

    模型懒加载——首次调用 ``embed`` / ``embed_query`` 时才载入权重，
    避免 import 阶段阻塞。

    Args:
        model_name: HuggingFace 模型名称，默认 ``BAAI/bge-large-zh-v1.5``。
        device: 推理设备；为 ``None`` 时自动检测（cuda > cpu）。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    # ── 属性 ─────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """返回模型输出的向量维度。"""
        if self._dimension is None:
            self._ensure_model()
            assert self._dimension is not None  # noqa: S101
        return self._dimension

    # ── 公有方法 ─────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量化。

        Args:
            texts: 待编码的文本列表。

        Returns:
            每条文本对应的浮点向量列表。
        """
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()  # type: ignore[union-attr]

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。

        Args:
            text: 查询文本。

        Returns:
            浮点向量。
        """
        return self.embed([text])[0]

    # ── 私有方法 ─────────────────────────────────────

    def _ensure_model(self) -> SentenceTransformer:
        """懒加载模型，首次调用时初始化。"""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        device = self._device or self._detect_device()
        logger.info(
            "embedder_loading",
            model=self._model_name,
            device=device,
        )
        self._model = SentenceTransformer(self._model_name, device=device)
        # 推断向量维度
        self._dimension = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info(
            "embedder_loaded",
            model=self._model_name,
            dimension=self._dimension,
        )
        return self._model

    @staticmethod
    def _detect_device() -> str:
        """自动检测可用设备。"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
