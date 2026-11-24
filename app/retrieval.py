import asyncio
import sys
from pathlib import Path

import psycopg
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg_pool import AsyncConnectionPool
from pypdf import PdfReader

from app.config import DATABASE_URL, EMBED_DIM, EMBED_MODEL

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30,
                                          length_function=lambda text: len(text.split()))

HYBRID_SQL = """
    WITH terms AS (
        SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery AS q
    ), semantic AS (
        SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank
        FROM chunks
        ORDER BY embedding <=> %(vector)s::vector
        LIMIT 20
    ), keyword AS (
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rank
        FROM chunks, terms
        WHERE tsv @@ q
        ORDER BY ts_rank_cd(tsv, q) DESC
        LIMIT 20
    )
    SELECT c.source, c.page, c.content,
           COALESCE(1.0 / (60 + s.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS score
    FROM semantic s
    FULL JOIN keyword k USING (id)
    JOIN chunks c USING (id)
    ORDER BY score DESC
    LIMIT %(limit)s
"""


def embeddings() -> Embeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)


def load_pdf(path: Path) -> list[Document]:
    pages = [Document(page.extract_text() or "", metadata={"source": path.name, "page": number})
             for number, page in enumerate(PdfReader(path).pages, 1)]
    return [chunk for chunk in SPLITTER.split_documents(pages) if chunk.page_content.strip()]


async def ingest(conn: psycopg.AsyncConnection, embedder: Embeddings, path: Path) -> int:
    chunks = load_pdf(path)
    vectors = await embedder.aembed_documents([chunk.page_content for chunk in chunks]) if chunks else []
    async with conn.transaction():
        await conn.execute("DELETE FROM chunks WHERE source = %s", (path.name,))
        async with conn.cursor() as cursor:
            await cursor.executemany(
                "INSERT INTO chunks (source, page, content, embedding) VALUES (%s, %s, %s, %s::vector)",
                [(chunk.metadata["source"], chunk.metadata["page"], chunk.page_content, str(vector))
                 for chunk, vector in zip(chunks, vectors)])
    return len(chunks)


async def ingest_folder(folder: Path, embedder: Embeddings) -> dict[str, int]:
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        return {pdf.name: await ingest(conn, embedder, pdf) for pdf in sorted(folder.glob("*.pdf"))}


class HybridRetriever(BaseRetriever):
    """pgvector cosine search and PostgreSQL full-text search, merged with reciprocal rank fusion."""

    pool: AsyncConnectionPool
    embedder: Embeddings
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        raise NotImplementedError("HybridRetriever is async only; use ainvoke")

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun):
        vector = await self.embedder.aembed_query(query)
        async with self.pool.connection() as conn:
            cursor = await conn.execute(HYBRID_SQL, {"vector": str(vector), "query": query, "limit": self.k})
            rows = await cursor.fetchall()
        return [Document(row["content"], metadata={"source": row["source"], "page": row["page"],
                                                   "score": float(row["score"])})
                for row in rows]


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    for name, count in asyncio.run(ingest_folder(folder, embeddings())).items():
        print(f"{name}: {count} chunks")
