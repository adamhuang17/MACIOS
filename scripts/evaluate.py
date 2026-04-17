"""RAG 评测 CLI 入口。

用法::

    python scripts/evaluate.py --dataset data/eval/qa.jsonl --output reports/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 确保 src 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_hub.config.settings import get_settings
from agent_hub.eval.dataset import EvalDataset
from agent_hub.eval.evaluator import RAGEvaluator
from agent_hub.eval.reporter import EvalReporter
from agent_hub.rag.pipeline import RAGPipeline


async def main(dataset_path: str, output_dir: str, top_k: int) -> None:
    """执行评测流程。"""
    # 加载数据集
    dataset = EvalDataset.from_jsonl(dataset_path)
    print(f"加载 {len(dataset)} 条评测样本")

    # 初始化 RAG
    settings = get_settings()
    rag = RAGPipeline(settings=settings)

    # 评测
    evaluator = RAGEvaluator(top_k=top_k)
    results = []

    for i, qa in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {qa.question[:50]}...")
        search_results = await rag.retrieve(
            query=qa.question,
            user_id="eval_user",
        )
        result = evaluator.evaluate_single(qa, search_results)
        results.append(result)

    # 汇总
    aggregate = evaluator.aggregate(results)

    # 报告
    reporter = EvalReporter()
    md_path, json_path = reporter.save(aggregate, output_dir)

    print(f"\n{'='*50}")
    print(f"Precision@{top_k}: {aggregate.avg_precision_at_k:.4f}")
    print(f"Recall@{top_k}:    {aggregate.avg_recall_at_k:.4f}")
    print(f"MRR:              {aggregate.mrr:.4f}")
    print(f"{'='*50}")
    print(f"报告已保存: {md_path}, {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--dataset", required=True, help="JSONL 评测数据集路径")
    parser.add_argument("--output", default="reports/", help="报告输出目录")
    parser.add_argument("--top-k", type=int, default=5, help="评估 top-K")
    args = parser.parse_args()

    asyncio.run(main(args.dataset, args.output, args.top_k))
