"""PostgreSQL/pgvector storage for enterprise RAG knowledge chunks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = structlog.get_logger(__name__)


@dataclass
class ChunkDoc:
    """A normalized document chunk ready for embedding and retrieval."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    document_id: str | None = None
    content_hash: str | None = None


@dataclass
class SearchResult:
    """A single retrieval result."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: int | None = None
    namespace: str | None = None
    document_id: str | None = None


class VectorStore:
    """PostgreSQL + pgvector canonical store.

    The store keeps document metadata in ``rag_documents`` and chunk vectors in
    ``rag_chunks``. Chunks are upserted idempotently by
    ``(user_id, namespace, document_id, chunk_index)`` so re-ingestion replaces
    the previous version instead of duplicating rows.
    """

    CHUNK_TABLE = "rag_chunks"
    DOCUMENT_TABLE = "rag_documents"
    # Compatibility with older code/tests.
    TABLE_NAME = CHUNK_TABLE

    def __init__(
        self,
        pg_dsn: str,
        dimension: int = 1024,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Vector dimension must be greater than zero")
        self._pg_dsn = pg_dsn
        self._dimension = dimension
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: AsyncConnectionPool | None = None
        self._schema_ready = False

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is not None:
            return self._pool

        import psycopg_pool

        self._pool = psycopg_pool.AsyncConnectionPool(
            self._pg_dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
            open=False,
        )
        await self._pool.open()
        return self._pool

    async def ensure_schema(self) -> None:
        """Create or migrate the pgvector schema."""
        if self._schema_ready:
            return
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.DOCUMENT_TABLE} (
                    id           BIGSERIAL PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    namespace    TEXT NOT NULL,
                    document_id  TEXT NOT NULL,
                    doc_type     TEXT NOT NULL DEFAULT 'text',
                    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    content_hash TEXT NOT NULL DEFAULT '',
                    chunk_count  INTEGER NOT NULL DEFAULT 0,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (user_id, namespace, document_id)
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.CHUNK_TABLE} (
                    id           BIGSERIAL PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    namespace    TEXT NOT NULL,
                    document_id  TEXT NOT NULL,
                    chunk_index  INTEGER NOT NULL DEFAULT 0,
                    content      TEXT NOT NULL,
                    content_hash TEXT NOT NULL DEFAULT '',
                    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding    vector({self._dimension}) NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # Migrate early skeleton tables in-place where possible.
            await conn.execute(
                f"ALTER TABLE {self.CHUNK_TABLE} ADD COLUMN IF NOT EXISTS document_id TEXT"
            )
            await conn.execute(
                f"""
                ALTER TABLE {self.CHUNK_TABLE}
                ADD COLUMN IF NOT EXISTS chunk_index INTEGER NOT NULL DEFAULT 0
                """
            )
            await conn.execute(
                f"""
                ALTER TABLE {self.CHUNK_TABLE}
                ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT ''
                """
            )
            await conn.execute(
                f"""
                ALTER TABLE {self.CHUNK_TABLE}
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                """
            )
            await conn.execute(
                f"UPDATE {self.CHUNK_TABLE} SET document_id = namespace WHERE document_id IS NULL"
            )
            await conn.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id, namespace, document_id
                            ORDER BY id
                        ) - 1 AS rn
                    FROM {self.CHUNK_TABLE}
                )
                UPDATE {self.CHUNK_TABLE} AS chunk
                SET chunk_index = ranked.rn
                FROM ranked
                WHERE chunk.id = ranked.id
                  AND chunk.chunk_index = 0
                """
            )
            await conn.execute(
                f"ALTER TABLE {self.CHUNK_TABLE} ALTER COLUMN document_id SET NOT NULL"
            )

            await conn.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_{self.CHUNK_TABLE}_uniq
                ON {self.CHUNK_TABLE} (user_id, namespace, document_id, chunk_index)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.CHUNK_TABLE}_hnsw
                ON {self.CHUNK_TABLE}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.CHUNK_TABLE}_user_ns
                ON {self.CHUNK_TABLE} (user_id, namespace)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.CHUNK_TABLE}_doc
                ON {self.CHUNK_TABLE} (user_id, namespace, document_id)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.CHUNK_TABLE}_metadata_gin
                ON {self.CHUNK_TABLE} USING gin (metadata)
                """
            )
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.DOCUMENT_TABLE}_user_ns
                ON {self.DOCUMENT_TABLE} (user_id, namespace)
                """
            )
            await conn.commit()
            logger.info(
                "vector_store_schema_ensured",
                chunk_table=self.CHUNK_TABLE,
                document_table=self.DOCUMENT_TABLE,
                dimension=self._dimension,
            )
            self._schema_ready = True

    async def upsert(
        self,
        user_id: str,
        namespace: str,
        chunks: list[ChunkDoc],
        embeddings: list[list[float]],
        *,
        document_id: str | None = None,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> int:
        """Compatibility wrapper for chunk upsert."""
        resolved_document_id = document_id
        if resolved_document_id is None and chunks:
            resolved_document_id = chunks[0].document_id or str(
                chunks[0].metadata.get("document_id", "")
            )
        return await self.upsert_document(
            user_id=user_id,
            namespace=namespace,
            document_id=resolved_document_id or namespace,
            chunks=chunks,
            embeddings=embeddings,
            doc_type=doc_type,
            metadata=metadata,
            replace=replace,
        )

    async def upsert_document(
        self,
        user_id: str,
        namespace: str,
        document_id: str,
        chunks: list[ChunkDoc],
        embeddings: list[list[float]],
        *,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> int:
        """Upsert one logical document and all of its embedded chunks."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        pool = await self._get_pool()
        metadata = metadata or {}
        doc_hash = self._hash_text("\n\n".join(chunk.content for chunk in chunks))

        async with pool.connection() as conn:
            if replace:
                await conn.execute(
                    f"""
                    DELETE FROM {self.CHUNK_TABLE}
                    WHERE user_id = %s AND namespace = %s AND document_id = %s
                    """,
                    (user_id, namespace, document_id),
                )

            await conn.execute(
                f"""
                INSERT INTO {self.DOCUMENT_TABLE}
                    (
                        user_id,
                        namespace,
                        document_id,
                        doc_type,
                        metadata,
                        content_hash,
                        chunk_count
                    )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (user_id, namespace, document_id)
                DO UPDATE SET
                    doc_type = EXCLUDED.doc_type,
                    metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash,
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at = NOW()
                """,
                (
                    user_id,
                    namespace,
                    document_id,
                    doc_type,
                    self._json(metadata),
                    doc_hash,
                    len(chunks),
                ),
            )

            async with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    self._validate_embedding(embedding)
                    chunk_doc_id = chunk.document_id or document_id
                    content_hash = chunk.content_hash or self._hash_text(chunk.content)
                    chunk_metadata = {
                        **metadata,
                        **chunk.metadata,
                        "namespace": namespace,
                        "document_id": chunk_doc_id,
                        "chunk_index": chunk.chunk_index,
                        "content_hash": content_hash,
                    }
                    await cur.execute(
                        f"""
                        INSERT INTO {self.CHUNK_TABLE}
                            (
                                user_id, namespace, document_id, chunk_index,
                                content, content_hash, metadata, embedding
                            )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (user_id, namespace, document_id, chunk_index)
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            content_hash = EXCLUDED.content_hash,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                        """,
                        (
                            user_id,
                            namespace,
                            chunk_doc_id,
                            chunk.chunk_index,
                            chunk.content,
                            content_hash,
                            self._json(chunk_metadata),
                            self._format_vector(embedding),
                        ),
                    )
            await conn.commit()

        logger.info(
            "vector_store_upsert",
            user_id=user_id,
            namespace=namespace,
            document_id=document_id,
            count=len(chunks),
            replace=replace,
        )
        return len(chunks)

    async def search_dense(
        self,
        query_embedding: list[float],
        user_id: str,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Dense cosine search through pgvector."""
        self._validate_embedding(query_embedding)
        pool = await self._get_pool()
        emb = self._format_vector(query_embedding)

        if namespace is None:
            sql = f"""
                SELECT id, content, metadata, namespace, document_id,
                       1 - (embedding <=> %(emb)s::vector) AS score
                FROM {self.CHUNK_TABLE}
                WHERE user_id = %(uid)s
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(topk)s
            """
            params: dict[str, Any] = {"uid": user_id, "emb": emb, "topk": top_k}
        else:
            sql = f"""
                SELECT id, content, metadata, namespace, document_id,
                       1 - (embedding <=> %(emb)s::vector) AS score
                FROM {self.CHUNK_TABLE}
                WHERE user_id = %(uid)s AND namespace = %(ns)s
                ORDER BY embedding <=> %(emb)s::vector
                LIMIT %(topk)s
            """
            params = {"uid": user_id, "ns": namespace, "emb": emb, "topk": top_k}

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [self._row_to_result(row) for row in rows]

    async def search_dense_raw(
        self,
        query_embedding: list[float],
        user_id: str,
        namespace: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Backward-compatible alias for ``search_dense``."""
        return await self.search_dense(query_embedding, user_id, namespace, top_k)

    async def list_chunks(
        self,
        user_id: str,
        namespace: str | None = None,
        *,
        document_id: str | None = None,
        limit: int = 20_000,
    ) -> list[ChunkDoc]:
        """Load chunks for sparse-index rebuilding."""
        pool = await self._get_pool()
        clauses = ["user_id = %(uid)s"]
        params: dict[str, Any] = {"uid": user_id, "limit": limit}
        if namespace is not None:
            clauses.append("namespace = %(ns)s")
            params["ns"] = namespace
        if document_id is not None:
            clauses.append("document_id = %(doc)s")
            params["doc"] = document_id

        where_clause = " AND ".join(clauses)
        sql = f"""
            SELECT id, content, metadata, namespace, document_id, chunk_index, content_hash
            FROM {self.CHUNK_TABLE}
            WHERE {where_clause}
            ORDER BY namespace, document_id, chunk_index
            LIMIT %(limit)s
        """
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

        chunks: list[ChunkDoc] = []
        for row in rows:
            metadata = self._parse_metadata(row[2])
            metadata.setdefault("chunk_id", row[0])
            metadata.setdefault("namespace", row[3])
            metadata.setdefault("document_id", row[4])
            metadata.setdefault("chunk_index", row[5])
            metadata.setdefault("content_hash", row[6])
            chunks.append(
                ChunkDoc(
                    content=row[1],
                    metadata=metadata,
                    chunk_index=int(row[5]),
                    document_id=row[4],
                    content_hash=row[6],
                )
            )
        return chunks

    async def list_namespaces(self, user_id: str) -> list[str]:
        """Return namespaces that contain chunks for one user."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT DISTINCT namespace
                FROM {self.CHUNK_TABLE}
                WHERE user_id = %s
                ORDER BY namespace
                """,
                (user_id,),
            )
            rows = await cur.fetchall()
        return [str(row[0]) for row in rows]

    async def delete_by_document(
        self,
        user_id: str,
        namespace: str,
        document_id: str,
    ) -> int:
        """Delete one logical document and its chunks."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    DELETE FROM {self.CHUNK_TABLE}
                    WHERE user_id = %s AND namespace = %s AND document_id = %s
                    """,
                    (user_id, namespace, document_id),
                )
                deleted = cur.rowcount or 0
                await cur.execute(
                    f"""
                    DELETE FROM {self.DOCUMENT_TABLE}
                    WHERE user_id = %s AND namespace = %s AND document_id = %s
                    """,
                    (user_id, namespace, document_id),
                )
            await conn.commit()
        logger.info(
            "vector_store_delete_document",
            user_id=user_id,
            namespace=namespace,
            document_id=document_id,
            deleted=deleted,
        )
        return deleted

    async def delete_by_namespace(self, user_id: str, namespace: str) -> int:
        """Delete all chunks/documents in a namespace."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {self.CHUNK_TABLE} WHERE user_id = %s AND namespace = %s",
                    (user_id, namespace),
                )
                deleted = cur.rowcount or 0
                await cur.execute(
                    f"DELETE FROM {self.DOCUMENT_TABLE} WHERE user_id = %s AND namespace = %s",
                    (user_id, namespace),
                )
            await conn.commit()
        logger.info(
            "vector_store_delete_namespace",
            user_id=user_id,
            namespace=namespace,
            deleted=deleted,
        )
        return deleted

    async def healthcheck(self) -> dict[str, Any]:
        """Check PostgreSQL connectivity and pgvector availability."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
            await cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            version_row = await cur.fetchone()
        return {
            "ok": True,
            "dimension": self._dimension,
            "pgvector": version_row[0] if version_row else None,
            "chunk_table": self.CHUNK_TABLE,
            "document_table": self.DOCUMENT_TABLE,
        }

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _validate_embedding(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {len(embedding)}"
            )

    @staticmethod
    def _format_vector(embedding: list[float]) -> str:
        return "[" + ",".join(f"{float(value):.12g}" for value in embedding) + "]"

    @staticmethod
    def _json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _parse_metadata(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if raw is None:
            return {}
        return json.loads(raw)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _row_to_result(cls, row: Any) -> SearchResult:
        metadata = cls._parse_metadata(row[2])
        metadata.setdefault("namespace", row[3])
        metadata.setdefault("document_id", row[4])
        metadata.setdefault("chunk_id", row[0])
        return SearchResult(
            content=row[1],
            score=float(row[5]),
            metadata=metadata,
            chunk_id=int(row[0]),
            namespace=row[3],
            document_id=row[4],
        )
