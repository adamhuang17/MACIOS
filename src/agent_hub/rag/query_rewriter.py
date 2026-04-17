"""Query 重写模块。

利用 LLM 将口语化用户查询重写为更适合向量检索的结构化语句。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from openai import OpenAI

logger = structlog.get_logger(__name__)

_REWRITE_PROMPT = """\
你是一个搜索查询优化专家。用户的原始查询可能是口语化的、模糊的或不完整的。
请将其重写为更精确、更适合知识库检索的查询语句。

规则：
1. 保留关键实体和术语
2. 移除口语化表达和语气词
3. 补充缺失的上下文信息（如果从上下文中可以推断）
4. 输出一句简洁的检索语句
5. 仅输出重写后的查询，不要有任何解释

原始查询：{query}
{context_section}
重写后的查询："""


class QueryRewriter:
    """LLM 驱动的 Query 重写器。

    Args:
        llm_client: OpenAI 兼容客户端。
        model: 模型名称。
    """

    def __init__(
        self,
        llm_client: OpenAI,
        model: str = "glm-4-flash",
    ) -> None:
        self._client = llm_client
        self._model = model

    async def rewrite(
        self,
        original_query: str,
        context: str = "",
    ) -> str:
        """将口语化查询重写为结构化检索语句。

        Args:
            original_query: 用户原始查询。
            context: 可选的上下文信息（如历史对话摘要）。

        Returns:
            重写后的查询字符串；失败时返回原始查询。
        """
        context_section = f"\n相关上下文：{context}\n" if context else ""
        prompt = _REWRITE_PROMPT.format(
            query=original_query,
            context_section=context_section,
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            rewritten = resp.choices[0].message.content.strip()
            if not rewritten:
                return original_query

            logger.info(
                "query_rewrite",
                original=original_query[:80],
                rewritten=rewritten[:80],
            )
            return rewritten

        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            return original_query
