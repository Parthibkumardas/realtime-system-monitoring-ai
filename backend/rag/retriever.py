"""
Sits between the raw vector store and the RAG pipeline: runs the query,
applies a similarity floor, and formats hits into a compact context block
plus a citation list the API can return to the frontend.
"""
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, similarity_floor: float = 0.3):
        self.vector_store = vector_store
        self.similarity_floor = similarity_floor

    def retrieve(self, query: str, n_results: int = 5) -> dict:
        hits = self.vector_store.query(query, n_results=n_results)
        hits = [h for h in hits if h["similarity"] >= self.similarity_floor]

        context_lines = []
        citations = []
        for h in hits:
            source = h["metadata"].get("source", "unknown")
            context_lines.append(f"[{source}] {h['text']}")
            citations.append(
                {
                    "source": source,
                    "similarity": h["similarity"],
                    "snippet": h["text"][:180],
                }
            )

        return {
            "context": "\n".join(context_lines),
            "citations": citations,
            "hit_count": len(hits),
        }
