"""QA 测试集管理。

支持 JSONL 格式的 QA 对加载/保存。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAPair:
    """一条评测 QA 对。"""

    question: str
    expected_answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EvalDataset:
    """JSONL 评测数据集。

    Args:
        pairs: QA 对列表。
    """

    def __init__(self, pairs: list[QAPair] | None = None) -> None:
        self.pairs = pairs or []

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[QAPair]:
        return iter(self.pairs)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalDataset:
        """从 JSONL 文件加载。

        每行格式::

            {"question": "...", "expected_answer": "...", "relevant_doc_ids": [...]}
        """
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pairs.append(QAPair(
                    question=obj["question"],
                    expected_answer=obj["expected_answer"],
                    relevant_doc_ids=obj.get("relevant_doc_ids", []),
                    metadata=obj.get("metadata", {}),
                ))
        return cls(pairs)

    def to_jsonl(self, path: str | Path) -> None:
        """保存为 JSONL 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            for p in self.pairs:
                obj = {
                    "question": p.question,
                    "expected_answer": p.expected_answer,
                    "relevant_doc_ids": p.relevant_doc_ids,
                    "metadata": p.metadata,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
