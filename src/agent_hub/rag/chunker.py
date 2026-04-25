"""文档语义切块。

支持动态窗口 + 重叠切分，按段落边界对齐。
支持 Markdown、纯文本和 PDF（PyMuPDF）。
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agent_hub.rag.vector_store import ChunkDoc

logger = structlog.get_logger(__name__)

# token 估算：中文约 1 字 = 1.5 token，英文约 4 字符 = 1 token
# 简化为 len(text) / 2 作为近似
_TOKEN_RATIO = 2


def _estimate_tokens(text: str) -> int:
    """粗估 token 数量。"""
    return max(1, len(text) // _TOKEN_RATIO)


class DocumentChunker:
    """文档语义切块器。

    Args:
        chunk_size: 目标切块大小（token 估算值）。
        chunk_overlap: 重叠大小（token 估算值）。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # 以字符计的近似目标
        self._char_size = chunk_size * _TOKEN_RATIO
        self._char_overlap = chunk_overlap * _TOKEN_RATIO

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkDoc]:
        """对纯文本进行切块（按段落边界对齐）。

        Args:
            text: 原始文本。
            metadata: 附加元数据。

        Returns:
            切块列表。
        """
        metadata = metadata or {}
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        return self._merge_paragraphs(paragraphs, metadata)

    def chunk_markdown(
        self,
        md_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkDoc]:
        """对 Markdown 文本进行切块（按标题分段后再切）。

        Args:
            md_text: Markdown 原文。
            metadata: 附加元数据。

        Returns:
            切块列表。
        """
        metadata = metadata or {}
        if not md_text.strip():
            return []

        sections = self._split_markdown_sections(md_text)
        chunks: list[ChunkDoc] = []
        for section_title, section_body in sections:
            section_meta = {**metadata}
            if section_title:
                section_meta["section"] = section_title
            paragraphs = self._split_paragraphs(section_body)
            chunks.extend(self._merge_paragraphs(paragraphs, section_meta))

        # 重新编号 chunk_index
        for i, c in enumerate(chunks):
            c.chunk_index = i

        return chunks

    def chunk_pdf(
        self,
        pdf_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkDoc]:
        """解析 PDF 文件并切块。

        Args:
            pdf_path: PDF 文件路径。
            metadata: 附加元数据。

        Returns:
            切块列表。
        """
        import fitz  # PyMuPDF

        metadata = metadata or {}
        doc = fitz.open(pdf_path)
        all_text_parts: list[str] = []

        for _page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if page_text.strip():
                all_text_parts.append(page_text)

        doc.close()

        if not all_text_parts:
            return []

        full_text = "\n\n".join(all_text_parts)
        pdf_meta = {**metadata, "source_type": "pdf", "source": pdf_path}
        return self.chunk_text(full_text, pdf_meta)

    # ── 私有方法 ─────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """按空行或双换行分段。"""
        raw = re.split(r"\n\s*\n", text)
        return [p.strip() for p in raw if p.strip()]

    @staticmethod
    def _split_markdown_sections(md_text: str) -> list[tuple[str, str]]:
        """按 Markdown 标题分段。

        Returns:
            ``(section_title, section_body)`` 元组列表。
        """
        # 匹配 # / ## / ### 等标题行
        pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections: list[tuple[str, str]] = []
        last_pos = 0
        last_title = ""

        for match in pattern.finditer(md_text):
            # 将上一段内容收入
            body = md_text[last_pos:match.start()].strip()
            if body or last_title:
                sections.append((last_title, body))
            last_title = match.group(2).strip()
            last_pos = match.end()

        # 最后一段
        body = md_text[last_pos:].strip()
        if body or last_title:
            sections.append((last_title, body))

        if not sections:
            sections.append(("", md_text.strip()))

        return sections

    def _merge_paragraphs(
        self,
        paragraphs: list[str],
        metadata: dict[str, Any],
    ) -> list[ChunkDoc]:
        """将段落合并为不超过目标大小的切块，保留重叠。"""
        chunks: list[ChunkDoc] = []
        current_parts: list[str] = []
        current_len = 0
        idx = 0

        for para in paragraphs:
            para_len = len(para)

            # 单段落超过 chunk_size，强制切分
            if para_len > self._char_size:
                # 先收集已有
                if current_parts:
                    chunks.append(ChunkDoc(
                        content="\n\n".join(current_parts),
                        metadata=dict(metadata),
                        chunk_index=idx,
                    ))
                    idx += 1
                    current_parts = []
                    current_len = 0

                # 对超长段落做硬切
                for sub in self._hard_split(para):
                    chunks.append(ChunkDoc(
                        content=sub,
                        metadata=dict(metadata),
                        chunk_index=idx,
                    ))
                    idx += 1
                continue

            if current_len + para_len > self._char_size and current_parts:
                # 达到目标大小，输出当前块
                chunks.append(ChunkDoc(
                    content="\n\n".join(current_parts),
                    metadata=dict(metadata),
                    chunk_index=idx,
                ))
                idx += 1

                # 保留重叠部分
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) > self._char_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)

                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(para)
            current_len += para_len

        # 最后剩余
        if current_parts:
            chunks.append(ChunkDoc(
                content="\n\n".join(current_parts),
                metadata=dict(metadata),
                chunk_index=idx,
            ))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """对超长文本进行硬切分（按句号/换行优先）。"""
        result: list[str] = []
        while len(text) > self._char_size:
            # 尝试在目标长度附近找句子边界
            cut_pos = self._char_size
            for sep in ["。", ".", "\n", "；", ";", "，", ","]:
                pos = text.rfind(sep, 0, self._char_size)
                if pos > self._char_size // 2:
                    cut_pos = pos + 1
                    break

            result.append(text[:cut_pos].strip())
            # 重叠
            overlap_start = max(0, cut_pos - self._char_overlap)
            text = text[overlap_start:]

        if text.strip():
            result.append(text.strip())

        return result
