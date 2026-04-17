"""评测模块测试。

覆盖 Dataset、Evaluator、Reporter。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_hub.eval.dataset import EvalDataset, QAPair
from agent_hub.eval.evaluator import AggregateResult, EvalResult, RAGEvaluator
from agent_hub.eval.reporter import EvalReporter
from agent_hub.rag.vector_store import SearchResult


# ═══════════════════════════════════════════════════════
# Dataset 测试
# ═══════════════════════════════════════════════════════


class TestEvalDataset:
    """QA 数据集测试。"""

    def test_from_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            '{"question":"Q1","expected_answer":"A1","relevant_doc_ids":["d1"]}\n'
            '{"question":"Q2","expected_answer":"A2","relevant_doc_ids":["d2","d3"]}\n',
            encoding="utf-8",
        )
        ds = EvalDataset.from_jsonl(jsonl)
        assert len(ds) == 2
        assert ds.pairs[0].question == "Q1"
        assert ds.pairs[1].relevant_doc_ids == ["d2", "d3"]

    def test_to_jsonl(self, tmp_path: Path) -> None:
        ds = EvalDataset([
            QAPair(question="Q1", expected_answer="A1", relevant_doc_ids=["d1"]),
        ])
        out = tmp_path / "out.jsonl"
        ds.to_jsonl(out)

        loaded = EvalDataset.from_jsonl(out)
        assert len(loaded) == 1
        assert loaded.pairs[0].question == "Q1"

    def test_empty_dataset(self) -> None:
        ds = EvalDataset()
        assert len(ds) == 0

    def test_iteration(self) -> None:
        ds = EvalDataset([
            QAPair(question="Q1", expected_answer="A1"),
            QAPair(question="Q2", expected_answer="A2"),
        ])
        questions = [p.question for p in ds]
        assert questions == ["Q1", "Q2"]


# ═══════════════════════════════════════════════════════
# Evaluator 测试
# ═══════════════════════════════════════════════════════


class TestRAGEvaluator:
    """RAG 评测器测试。"""

    def _make_result(self, doc_id: str, score: float = 0.9) -> SearchResult:
        return SearchResult(
            chunk_id=doc_id,
            content="content",
            score=score,
            metadata={"doc_id": doc_id},
        )

    def test_precision_all_relevant(self) -> None:
        evaluator = RAGEvaluator(top_k=3)
        qa = QAPair(
            question="test",
            expected_answer="answer",
            relevant_doc_ids=["d1", "d2", "d3"],
        )
        results = [self._make_result(f"d{i}") for i in range(1, 4)]
        r = evaluator.evaluate_single(qa, results)
        assert r.precision_at_k == 1.0

    def test_precision_none_relevant(self) -> None:
        evaluator = RAGEvaluator(top_k=3)
        qa = QAPair(
            question="test", expected_answer="answer",
            relevant_doc_ids=["d10"],
        )
        results = [self._make_result(f"d{i}") for i in range(1, 4)]
        r = evaluator.evaluate_single(qa, results)
        assert r.precision_at_k == 0.0

    def test_recall_partial(self) -> None:
        evaluator = RAGEvaluator(top_k=5)
        qa = QAPair(
            question="test", expected_answer="answer",
            relevant_doc_ids=["d1", "d2", "d3", "d4"],
        )
        results = [self._make_result("d1"), self._make_result("d3")]
        r = evaluator.evaluate_single(qa, results)
        assert r.recall_at_k == 0.5

    def test_recall_no_relevant_docs(self) -> None:
        evaluator = RAGEvaluator(top_k=5)
        qa = QAPair(question="test", expected_answer="answer", relevant_doc_ids=[])
        results = [self._make_result("d1")]
        r = evaluator.evaluate_single(qa, results)
        assert r.recall_at_k == 1.0

    def test_mrr_first_position(self) -> None:
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1"}
        rr = RAGEvaluator._reciprocal_rank(retrieved, relevant)
        assert rr == 1.0

    def test_mrr_third_position(self) -> None:
        retrieved = ["d2", "d3", "d1"]
        relevant = {"d1"}
        rr = RAGEvaluator._reciprocal_rank(retrieved, relevant)
        assert abs(rr - 1 / 3) < 1e-6

    def test_mrr_not_found(self) -> None:
        retrieved = ["d2", "d3"]
        relevant = {"d1"}
        rr = RAGEvaluator._reciprocal_rank(retrieved, relevant)
        assert rr == 0.0

    def test_aggregate_empty(self) -> None:
        evaluator = RAGEvaluator()
        agg = evaluator.aggregate([])
        assert agg.num_queries == 0

    def test_aggregate_multiple(self) -> None:
        evaluator = RAGEvaluator()
        results = [
            EvalResult(
                question="Q1", retrieved_ids=["d1"],
                relevant_ids=["d1"], precision_at_k=1.0,
                recall_at_k=1.0, reciprocal_rank=1.0,
            ),
            EvalResult(
                question="Q2", retrieved_ids=["d2"],
                relevant_ids=["d3"], precision_at_k=0.0,
                recall_at_k=0.0, reciprocal_rank=0.0,
            ),
        ]
        agg = evaluator.aggregate(results)
        assert agg.num_queries == 2
        assert agg.avg_precision_at_k == 0.5
        assert agg.mrr == 0.5


# ═══════════════════════════════════════════════════════
# Reporter 测试
# ═══════════════════════════════════════════════════════


class TestEvalReporter:
    """报告生成器测试。"""

    def _make_aggregate(self) -> AggregateResult:
        return AggregateResult(
            num_queries=2,
            avg_precision_at_k=0.75,
            avg_recall_at_k=0.5,
            mrr=0.8,
            per_query=[
                EvalResult(
                    question="Q1", retrieved_ids=["d1"],
                    relevant_ids=["d1"], precision_at_k=1.0,
                    recall_at_k=1.0, reciprocal_rank=1.0,
                ),
                EvalResult(
                    question="Q2", retrieved_ids=["d2"],
                    relevant_ids=["d3"], precision_at_k=0.5,
                    recall_at_k=0.0, reciprocal_rank=0.6,
                ),
            ],
        )

    def test_to_markdown(self) -> None:
        md = EvalReporter.to_markdown(self._make_aggregate())
        assert "Precision@K" in md
        assert "0.7500" in md
        assert "Q1" in md

    def test_to_json(self) -> None:
        j = EvalReporter.to_json(self._make_aggregate())
        data = json.loads(j)
        assert data["num_queries"] == 2
        assert len(data["per_query"]) == 2

    def test_save(self, tmp_path: Path) -> None:
        agg = self._make_aggregate()
        md_path, json_path = EvalReporter.save(agg, tmp_path)
        assert md_path.exists()
        assert json_path.exists()
        assert md_path.suffix == ".md"
        assert json_path.suffix == ".json"
