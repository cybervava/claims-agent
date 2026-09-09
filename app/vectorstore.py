"""ChromaDB-backed policy knowledge base. Embeddings are computed via LLMClient
(Azure text-embedding-3-small) so they are metered like every other call."""
import hashlib

import chromadb

from .config import Settings
from .db import Database, new_id
from .ingest import chunk_text, extract_text
from .llm import LLMClient
from .schemas import IngestResponse, SearchHit


class PolicyStore:
    def __init__(self, settings: Settings, llm: LLMClient, db: Database):
        self.settings = settings
        self.llm = llm
        self.db = db
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"})

    def ingest(self, filename: str, data: bytes, metadata: dict | None, actor: str) -> IngestResponse:
        sha = hashlib.sha256(data).hexdigest()
        existing = self.db.find_document_by_hash(sha)
        if existing:
            return IngestResponse(doc_id=existing["doc_id"], filename=existing["filename"],
                                  chunk_count=existing["chunk_count"], sha256=sha, skipped_duplicate=True)
        text = extract_text(filename, data)
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(f"no text could be extracted from {filename}")
        doc_id = new_id("doc")
        base_meta = {k: v for k, v in (metadata or {}).items() if v is not None}
        vectors = self.llm.embed(chunks, purpose="embed_policy")
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        metas = [{**base_meta, "doc_id": doc_id, "filename": filename, "chunk_index": i} for i in range(len(chunks))]
        self.collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metas)
        self.db.add_document(doc_id, filename, sha, len(chunks), self.settings.embedding_deployment,
                             base_meta, actor)
        return IngestResponse(doc_id=doc_id, filename=filename, chunk_count=len(chunks), sha256=sha)

    def search(self, query: str, top_k: int | None = None, where: dict | None = None,
               run_id: str | None = None) -> list[SearchHit]:
        if self.collection.count() == 0:
            return []
        k = min(top_k or self.settings.retrieval_top_k, self.collection.count())
        [vector] = self.llm.embed([query], run_id=run_id, purpose="embed_query")
        res = self.collection.query(query_embeddings=[vector], n_results=k, where=where or None,
                                    include=["documents", "metadatas", "distances"])
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(SearchHit(chunk_id=cid, doc_id=meta.get("doc_id", ""), filename=meta.get("filename", ""),
                                  score=round(1.0 - float(dist), 4), text=doc, metadata=meta))
        return hits

    def chunks(self, doc_id: str, include_embeddings: bool = False, preview_dims: int = 8) -> list[dict]:
        include = ["documents", "metadatas"] + (["embeddings"] if include_embeddings else [])
        res = self.collection.get(where={"doc_id": doc_id}, include=include)
        embeddings = res.get("embeddings") if include_embeddings else None
        rows = []
        for i, (cid, doc, meta) in enumerate(zip(res["ids"], res["documents"], res["metadatas"])):
            row = {"chunk_id": cid, "text": doc, "metadata": meta}
            if embeddings is not None:
                vec = [float(x) for x in embeddings[i]]
                row["embedding_dims"] = len(vec)
                row["embedding_preview"] = [round(x, 5) for x in vec[:preview_dims]]
            rows.append(row)
        return sorted(rows, key=lambda r: r["metadata"].get("chunk_index", 0))

    def delete(self, doc_id: str) -> bool:
        self.collection.delete(where={"doc_id": doc_id})
        return self.db.delete_document(doc_id)

    def stats(self) -> dict:
        return {"collection": self.settings.chroma_collection, "chunks": self.collection.count(),
                "documents": len(self.db.list_documents()),
                "embedding_model": self.settings.embedding_deployment}
