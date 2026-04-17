#!/usr/bin/env python3
"""RAG 检索评测数据集构建与评测。

用于生成 QA 数据集并运行 Precision@K / Recall@K / MRR 评测。
需要 PostgreSQL + pgvector 运行。

用法::

    # Step 1: 构建测试数据集
    python scripts/bench_rag.py build-dataset

    # Step 2: 入库测试文档（需要 pgvector）
    python scripts/bench_rag.py ingest

    # Step 3: 运行评测（需要 pgvector）
    python scripts/bench_rag.py evaluate
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── 内置测试文档 ────────────────────────────────────────

TEST_DOCUMENTS: list[dict] = [
    {
        "doc_id": "rag_overview",
        "content": """# RAG 技术概述

## 什么是 RAG

RAG（Retrieval-Augmented Generation）是一种将信息检索与文本生成相结合的技术架构。它通过在生成回答前先检索相关知识，来增强大语言模型的回答质量和准确性。

## RAG 的核心组件

RAG 系统通常包含三个核心组件：
1. 文档处理与索引：将文档切分为小块，生成向量表示并建立索引
2. 检索模块：根据用户查询检索最相关的文档片段
3. 生成模块：将检索到的上下文与用户问题一起输入 LLM，生成最终回答

## 向量检索原理

向量检索通过计算查询向量与文档向量之间的相似度来排序。常用的相似度度量包括余弦相似度、欧氏距离和内积。HNSW（Hierarchical Navigable Small World）是一种常用的高效近似最近邻搜索算法。

## 混合检索策略

单纯的向量检索在处理专业术语和精确匹配时可能存在不足。混合检索结合了稠密检索（向量）和稀疏检索（BM25），通过 RRF（Reciprocal Rank Fusion）融合排序，可以有效提升检索质量。
""",
        "namespace": "tech_docs",
    },
    {
        "doc_id": "python_best_practices",
        "content": """# Python 最佳实践

## 类型注解

Python 3.5+ 引入了类型注解，建议在函数签名中使用类型提示提高代码可读性和 IDE 支持。

## 异步编程

Python 的 asyncio 模块提供了异步编程支持。使用 async/await 语法可以编写高效的异步 I/O 程序。asyncio.gather 可以并发执行多个协程，asyncio.Semaphore 用于控制并发数量。

## 数据模型

推荐使用 Pydantic 进行数据验证和序列化。Pydantic V2 基于 Rust 实现，性能大幅提升。Pydantic 的 model_json_schema 方法可以直接生成 JSON Schema，方便与 OpenAI Function Calling 集成。

## 测试策略

Python 测试推荐使用 pytest 框架。结合 pytest-asyncio 可以测试异步代码。使用 fixture 管理测试依赖，mock 外部服务调用。覆盖率工具推荐 pytest-cov。
""",
        "namespace": "tech_docs",
    },
    {
        "doc_id": "db_indexing",
        "content": """# 数据库索引原理

## B+ 树索引

B+ 树是最常用的数据库索引结构。它的特点是所有数据都存储在叶子节点，非叶子节点只存储键值用于路由。B+ 树的范围查询效率很高，因为叶子节点通过链表连接。

## 向量索引

随着 AI 应用的发展，向量索引变得越来越重要。pgvector 扩展为 PostgreSQL 添加了向量相似度搜索能力。HNSW 索引是 pgvector 支持的主要索引类型，参数包括 m（连接数）和 ef_construction（构建时搜索宽度）。

## 索引优化

索引优化需要考虑以下因素：
- 查询频率：频繁查询的字段应该建立索引
- 数据区分度：低区分度的字段（如性别）不适合单独建索引
- 写入频率：索引会降低写入性能，需要权衡读写比例
- 复合索引：遵循最左前缀匹配原则
""",
        "namespace": "tech_docs",
    },
    {
        "doc_id": "microservice_arch",
        "content": """# 微服务架构设计

## 服务拆分原则

微服务拆分应遵循单一职责原则，每个服务专注于一个业务领域。服务间通过 API 或消息队列通信，避免共享数据库。

## 服务发现与注册

在微服务架构中，服务实例的地址是动态的。服务注册与发现机制（如 Consul、Etcd）可以让服务消费者动态找到可用的服务提供者。

## 分布式追踪

在微服务环境中，一个请求可能经过多个服务。分布式追踪系统（如 Jaeger、Zipkin）通过 Trace ID 将跨服务的调用链串联起来，便于排查性能瓶颈和错误。OpenTelemetry 是一个可观测性的开放标准。

## 容错与降级

微服务需要设计容错机制：
- 熔断器模式：当下游服务异常时自动切断请求
- 超时控制：设置合理的请求超时时间
- 限流：使用 Semaphore 或令牌桶控制请求速率
- 降级策略：当主要服务不可用时返回兜底结果
""",
        "namespace": "tech_docs",
    },
]


# ── QA 数据集 ──────────────────────────────────────────

QA_DATASET: list[dict] = [
    {
        "question": "RAG 是什么技术？",
        "relevant_doc_ids": ["rag_overview"],
        "tags": ["definition"],
    },
    {
        "question": "RAG 系统包含哪些核心组件？",
        "relevant_doc_ids": ["rag_overview"],
        "tags": ["component"],
    },
    {
        "question": "向量检索的原理是什么？",
        "relevant_doc_ids": ["rag_overview"],
        "tags": ["principle"],
    },
    {
        "question": "什么是混合检索？RRF 融合排序是怎么工作的？",
        "relevant_doc_ids": ["rag_overview"],
        "tags": ["hybrid"],
    },
    {
        "question": "Python 中如何使用异步编程？",
        "relevant_doc_ids": ["python_best_practices"],
        "tags": ["async"],
    },
    {
        "question": "Pydantic 的 model_json_schema 方法有什么用？",
        "relevant_doc_ids": ["python_best_practices"],
        "tags": ["pydantic"],
    },
    {
        "question": "pytest 怎么测试异步代码？",
        "relevant_doc_ids": ["python_best_practices"],
        "tags": ["testing"],
    },
    {
        "question": "B+ 树索引的特点是什么？",
        "relevant_doc_ids": ["db_indexing"],
        "tags": ["btree"],
    },
    {
        "question": "pgvector 的 HNSW 索引有哪些参数？",
        "relevant_doc_ids": ["db_indexing", "rag_overview"],
        "tags": ["pgvector"],
    },
    {
        "question": "数据库索引优化需要考虑哪些因素？",
        "relevant_doc_ids": ["db_indexing"],
        "tags": ["optimization"],
    },
    {
        "question": "微服务架构中如何做服务拆分？",
        "relevant_doc_ids": ["microservice_arch"],
        "tags": ["design"],
    },
    {
        "question": "什么是分布式追踪？OpenTelemetry 是什么？",
        "relevant_doc_ids": ["microservice_arch", "rag_overview"],
        "tags": ["tracing"],
    },
    {
        "question": "微服务的容错机制有哪些？",
        "relevant_doc_ids": ["microservice_arch"],
        "tags": ["fault_tolerance"],
    },
    {
        "question": "asyncio.Semaphore 的作用是什么？",
        "relevant_doc_ids": ["python_best_practices", "microservice_arch"],
        "tags": ["concurrency"],
    },
    {
        "question": "HNSW 算法是如何工作的？",
        "relevant_doc_ids": ["db_indexing", "rag_overview"],
        "tags": ["algorithm"],
    },
]


def build_dataset() -> None:
    """生成 JSONL 格式的 QA 数据集。"""
    output_dir = _ROOT / "data" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "qa_dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for qa in QA_DATASET:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"QA 数据集已生成: {dataset_path}")
    print(f"  样本数量: {len(QA_DATASET)}")

    # 同时生成测试文档
    docs_dir = output_dir / "test_docs"
    docs_dir.mkdir(exist_ok=True)
    for doc in TEST_DOCUMENTS:
        doc_path = docs_dir / f"{doc['doc_id']}.md"
        doc_path.write_text(doc["content"], encoding="utf-8")
    print(f"  测试文档已生成: {docs_dir}")


async def ingest() -> None:
    """入库测试文档。"""
    from agent_hub.rag.pipeline import RAGPipeline

    from agent_hub.config.settings import get_settings

    settings = get_settings()
    rag = RAGPipeline(settings=settings)

    user_id = "eval_user"
    total_chunks = 0

    for doc in TEST_DOCUMENTS:
        count = await rag.ingest(
            user_id=user_id,
            namespace=doc["namespace"],
            content=doc["content"],
            doc_type="markdown",
            metadata={"doc_id": doc["doc_id"]},
        )
        total_chunks += count
        print(f"  入库 {doc['doc_id']}: {count} chunks")

    await rag.close()
    print(f"  总入库 chunks: {total_chunks}")


async def evaluate() -> None:
    """运行评测。"""
    from agent_hub.eval.dataset import EvalDataset, QAPair
    from agent_hub.eval.evaluator import RAGEvaluator
    from agent_hub.eval.reporter import EvalReporter
    from agent_hub.rag.pipeline import RAGPipeline

    from agent_hub.config.settings import get_settings

    settings = get_settings()
    rag = RAGPipeline(settings=settings)

    # 加载数据集
    dataset_path = _ROOT / "data" / "eval" / "qa_dataset.jsonl"
    if not dataset_path.exists():
        print("数据集不存在，请先运行: python scripts/bench_rag.py build-dataset")
        return

    dataset = EvalDataset.from_jsonl(str(dataset_path))
    print(f"加载 {len(dataset)} 条评测样本")

    # 评测
    evaluator = RAGEvaluator(top_k=5)
    results = []

    for i, qa in enumerate(dataset, start=1):
        print(f"  [{i}/{len(dataset)}] {qa.question[:40]}...")
        search_results = await rag.retrieve(
            query=qa.question,
            user_id="eval_user",
        )
        result = evaluator.evaluate_single(qa, search_results)
        results.append(result)

    # 汇总
    aggregate = evaluator.aggregate(results)

    print(f"\n{'=' * 50}")
    print(f"评测结果 ({len(dataset)} 条样本)")
    print(f"{'=' * 50}")
    print(f"  Precision@5: {aggregate.avg_precision_at_k:.4f} ({aggregate.avg_precision_at_k:.0%})")
    print(f"  Recall@5:    {aggregate.avg_recall_at_k:.4f} ({aggregate.avg_recall_at_k:.0%})")
    print(f"  MRR:         {aggregate.mrr:.4f}")

    # 保存报告
    reporter = EvalReporter()
    output_dir = str(_ROOT / "reports")
    md_path, json_path = reporter.save(aggregate, output_dir)
    print(f"  报告已保存: {md_path}")

    await rag.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/bench_rag.py build-dataset  # 构建 QA 数据集")
        print("  python scripts/bench_rag.py ingest         # 入库测试文档")
        print("  python scripts/bench_rag.py evaluate       # 运行评测")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build-dataset":
        build_dataset()
    elif cmd == "ingest":
        asyncio.run(ingest())
    elif cmd == "evaluate":
        asyncio.run(evaluate())
    else:
        print(f"Unknown command: {cmd}")
