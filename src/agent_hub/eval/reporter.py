"""评测报告生成器。

输出 Markdown 和 JSON 两种格式。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_hub.eval.evaluator import AggregateResult


class EvalReporter:
    """评测报告生成器。"""

    @staticmethod
    def to_markdown(result: AggregateResult, title: str = "RAG 评测报告") -> str:
        """生成 Markdown 格式报告。"""
        lines = [
            f"# {title}",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**评测样本数**: {result.num_queries}",
            f"",
            f"## 汇总指标",
            f"",
            f"| 指标 | 值 |",
            f"|------|------|",
            f"| Precision@K | {result.avg_precision_at_k:.4f} |",
            f"| Recall@K | {result.avg_recall_at_k:.4f} |",
            f"| MRR | {result.mrr:.4f} |",
            f"",
        ]

        if result.per_query:
            lines.extend([
                f"## 逐条详情",
                f"",
                f"| # | 问题 | P@K | R@K | RR |",
                f"|---|------|-----|-----|----|",
            ])
            for i, r in enumerate(result.per_query, start=1):
                q = r.question[:30] + "..." if len(r.question) > 30 else r.question
                lines.append(
                    f"| {i} | {q} | {r.precision_at_k:.3f} | "
                    f"{r.recall_at_k:.3f} | {r.reciprocal_rank:.3f} |"
                )

        return "\n".join(lines)

    @staticmethod
    def to_json(result: AggregateResult) -> str:
        """生成 JSON 格式报告。"""
        data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": result.num_queries,
            "avg_precision_at_k": result.avg_precision_at_k,
            "avg_recall_at_k": result.avg_recall_at_k,
            "mrr": result.mrr,
            "per_query": [
                {
                    "question": r.question,
                    "retrieved_ids": r.retrieved_ids,
                    "relevant_ids": r.relevant_ids,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                }
                for r in result.per_query
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def save(
        result: AggregateResult,
        output_dir: str | Path,
        prefix: str = "eval",
    ) -> tuple[Path, Path]:
        """保存报告到文件。

        Returns:
            (markdown_path, json_path) 元组。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = output_dir / f"{prefix}_{ts}.md"
        json_path = output_dir / f"{prefix}_{ts}.json"

        md_path.write_text(
            EvalReporter.to_markdown(result), encoding="utf-8",
        )
        json_path.write_text(
            EvalReporter.to_json(result), encoding="utf-8",
        )

        return md_path, json_path
