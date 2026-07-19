"""
Persistent Chroma vector store for incident reports / runbooks / logs.
Seeded at startup from data/incidents.json (see main.py's `seed_vector_store`).
"""
import chromadb

from rag.embeddings import EmbeddingModel


class VectorStore:
    def __init__(self, persist_dir: str, embedder: EmbeddingModel, collection_name: str = "incidents"):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def add_documents(self, docs: list[dict]) -> None:
        """
        docs: list of {id, text, metadata} dicts.
        metadata should include at least: source, timestamp, service.
        """
        embeddings = self.embedder.embed([d["text"] for d in docs])
        self.collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
            embeddings=embeddings,
        )

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        query_embedding = self.embedder.embed_one(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = round(1 - dist, 3)  # cosine distance -> similarity
            hits.append({"text": doc, "metadata": meta, "similarity": similarity})
        return hits
