"""检索结果相关性评估模块。

利用 LLM 对检索结果的相关性进行 1-5 分打分。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from openai import OpenAI

    from agent_hub.rag.vector_store import SearchResult

logger = structlog.get_logger(__name__)

_EVAL_PROMPT = """\
你是一个检索质量评估专家。请评估以下检索结果与用户查询的相关性。

用户查询：{query}

检索结果：
{results_text}

评分标准（1-5 分）：
1分：完全不相关
2分：主题相关但无法回答查询
3分：部分相关，能提供一些有用信息
4分：高度相关，能基本回答查询
5分：完全相关，能完整回答查询

请只输出一个整数评分（1-5），不要有任何解释。
评分："""


class RelevanceEvaluator:
    """LLM 驱动的相关性评估器。

    Args:
        llm_client: OpenAI 兼容客户端。
        model: 模型名称。
        threshold: 及格分数阈值（≥ 此分数视为相关）。
    """

    def __init__(
        self,
        llm_client: OpenAI,
        model: str = "glm-4-flash",
        threshold: float = 3.0,
    ) -> None:
        self._client = llm_client
        self._model = model
        self.threshold = threshold

    async def evaluate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> float:
        """评估检索结果与查询的相关性。

        Args:
            query: 用户查询。
            results: 检索结果列表。

        Returns:
            相关性评分 (1.0-5.0)；异常时返回 3.0（中性分数，触发降级而非崩溃）。
        """
        if not results:
            return 1.0

        results_text = self._format_results(results)
        prompt = _EVAL_PROMPT.format(query=query, results_text=results_text)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            raw = resp.choices[0].message.content.strip()
            score = self._parse_score(raw)

            logger.info(
                "relevance_eval",
                query=query[:50],
                score=score,
                result_count=len(results),
            )
            return score

        except Exception as exc:
            logger.warning("relevance_eval_failed", error=str(exc))
            return 3.0

    @staticmethod
    def _format_results(results: list[SearchResult], max_chars: int = 2000) -> str:
        """将检索结果格式化为评估文本。"""
        parts: list[str] = []
        total = 0
        for i, r in enumerate(results, start=1):
            snippet = r.content[:300]
            part = f"[{i}] {snippet}"
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n\n".join(parts)

    @staticmethod
    def _parse_score(raw: str) -> float:
        """从 LLM 输出中解析评分。"""
        match = re.search(r"[1-5]", raw)
        if match:
            return float(match.group())
        return 3.0
