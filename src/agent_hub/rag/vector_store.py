"""pgvector 向量存储抽象层。

HNSW 索引，按 user_id / namespace 分区，CRUD 操作。
使用 psycopg 异步连接池。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = structlog.get_logger(__name__)


@dataclass
class ChunkDoc:
    """文档切块。"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0


@dataclass
class SearchResult:
    """检索结果。"""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: int | None = None


class VectorStore:
    """pgvector 向量存储层。

    Args:
        pg_dsn: PostgreSQL 连接字符串。
        dimension: 向量维度（从 Embedder 获取）。
    """

    TABLE_NAME = "rag_chunks"

    def __init__(self, pg_dsn: str, dimension: int = 1024) -> None:
        self._pg_dsn = pg_dsn
        self._dimension = dimension
        self._pool: AsyncConnectionPool | None = None

    async def _get_pool(self) -> AsyncConnectionPool:
        """获取或创建连接池。"""
        if self._pool is not None:
            return self._pool

        import psycopg_pool

        self._pool = psycopg_pool.AsyncConnectionPool(
            self._pg_dsn,
            min_size=2,
            max_size=10,
            open=False,
        )
        await self._pool.open()
        return self._pool

    async def ensure_schema(self) -> None:
        """创建表和 HNSW 索引（幂等操作）。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    id          SERIAL PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    namespace   TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    JSONB DEFAULT '{{}}'::jsonb,
                    embedding   vector({self._dimension}),
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # HNSW 索引
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_hnsw
                ON {self.TABLE_NAME}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
            """)
            # 组合 B-tree 索引用于分区过滤
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_user_ns
                ON {self.TABLE_NAME} (user_id, namespace)
            """)
            await conn.commit()
            logger.info("vector_store_schema_ensured", table=self.TABLE_NAME)

    async def upsert(
        self,
        user_id: str,
        namespace: str,
        chunks: list[ChunkDoc],
        embeddings: list[list[float]],
    ) -> int:
        """批量入库文档切块（连同向量）。

        Args:
            user_id: 用户 ID。
            namespace: 命名空间（文档名 / 来源标识）。
            chunks: 文档切块列表。
            embeddings: 对应的向量列表。

        Returns:
            插入的行数。
        """
        if not chunks:
            return 0

        pool = await self._get_pool()
        import json
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings, strict=False):
                    await cur.execute(
                        f"""
                        INSERT INTO {self.TABLE_NAME}
                            (user_id, namespace, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            namespace,
                            chunk.content,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            str(emb),
                        ),
                    )
            await conn.commit()

        logger.info(
            "vector_store_upsert",
            user_id=user_id,
            namespace=namespace,
            count=len(chunks),
        )
        return len(chunks)

    async def search_dense(
        self,
        query_embedding: list[float],
        user_id: str,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Dense（余弦相似度）检索。

        Args:
            query_embedding: 查询向量。
            user_id: 用户 ID。
            namespace: 命名空间（None 表示搜索该用户所有空间）。
            top_k: 返回的最大结果数。

        Returns:
            按相似度降序排列的 SearchResult 列表。
        """
        pool = await self._get_pool()
        import json

        where_clause = "WHERE user_id = %s"
        params: list[Any] = [user_id]
        if namespace is not None:
            where_clause += " AND namespace = %s"
            params.append(namespace)

        params.append(str(query_embedding))
        params.append(top_k)

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                    SELECT id, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self.TABLE_NAME}
                    {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                (*params[:-2], params[-2], params[-2], params[-1]),
            )
            rows = await cur.fetchall()

        # 修正参数顺序：WHERE 子句参数 + 两次 embedding 参数 + limit
        # 重写为更清晰的拼接
        results: list[SearchResult] = []
        for row in rows:
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
            results.append(SearchResult(
                content=row[1],
                score=float(row[3]),
                metadata=meta,
                chunk_id=row[0],
            ))
        return results

    async def search_dense_raw(
        self,
        query_embedding: list[float],
        user_id: str,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Dense 检索（更清晰的参数绑定实现）。"""
        pool = await self._get_pool()
        import json

        emb_str = str(query_embedding)

        if namespace is not None:
            sql = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> %(emb)s::vector) AS score
                FROM {self.TABLE_NAME}
                WHERE user_id = %(uid)s AND namespace = %(ns)s
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(topk)s
            """
            params_dict = {
                "uid": user_id,
                "ns": namespace,
                "emb": emb_str,
                "topk": top_k,
            }
        else:
            sql = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> %(emb)s::vector) AS score
                FROM {self.TABLE_NAME}
                WHERE user_id = %(uid)s
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(topk)s
            """
            params_dict = {
                "uid": user_id,
                "emb": emb_str,
                "topk": top_k,
            }

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params_dict)
            rows = await cur.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
            results.append(SearchResult(
                content=row[1],
                score=float(row[3]),
                metadata=meta,
                chunk_id=row[0],
            ))
        return results

    async def delete_by_namespace(self, user_id: str, namespace: str) -> int:
        """按命名空间删除所有切块。

        Args:
            user_id: 用户 ID。
            namespace: 命名空间。

        Returns:
            删除的行数。
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.TABLE_NAME} WHERE user_id = %s AND namespace = %s",
                    (user_id, namespace),
                )
                deleted = cur.rowcount
            await conn.commit()
        logger.info(
            "vector_store_delete",
            user_id=user_id,
            namespace=namespace,
            deleted=deleted,
        )
        return deleted or 0

    async def close(self) -> None:
        """关闭连接池。"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
