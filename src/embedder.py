from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer


class ChunkEmbedder:
    """Эмбеддинги чанков. Для моделей E5 к документам добавляется префикс `passage: `."""

    def __init__(self, model_name: str, device: str = "cpu", passage_prefix: str = "passage: "):
        print(f"[Embedder] Loading model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.passage_prefix = passage_prefix
        self.dimensions = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded. Dimensions: {self.dimensions}")

    def embed_documents(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        texts = [self.passage_prefix + doc["text"] for doc in documents]

        print(f"[Embedder] Encoding {len(texts)} chunks (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        results = []
        for doc, embedding in zip(documents, embeddings):
            results.append({
                "text": doc["text"],  # исходный текст без префикса
                "embedding": embedding.tolist(),
                "metadata": {
                    **doc.get("metadata", {}),
                    "embedding_model": self.model_name,
                    "embedding_dimensions": self.dimensions,
                },
            })
        return results
