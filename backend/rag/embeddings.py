"""
Thin wrapper around sentence-transformers so the rest of the RAG code
doesn't need to know which embedding model/library is in use.
"""
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
