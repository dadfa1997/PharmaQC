"""Этап 3: расчёт эмбеддингов чанков -> data/embeddings/embeddings.jsonl."""
import json
import os

from src.config_loader import ConfigLoader
from src.embedder import ChunkEmbedder


def run_embedding(config: ConfigLoader = None):
    config = config or ConfigLoader()
    emb_cfg = config.section("embedding")
    paths = config.section("paths")

    documents = []
    with open(paths["chunks_jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    print(f"[Embed] Loaded {len(documents)} chunks from {paths['chunks_jsonl']}")

    if not documents:
        print("[Embed] No chunks found. Run chunk.py first.")
        return

    embedder = ChunkEmbedder(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "cpu"),
        passage_prefix=emb_cfg.get("passage_prefix", "passage: "),
    )
    results = embedder.embed_documents(documents, batch_size=emb_cfg.get("batch_size", 32))

    os.makedirs(os.path.dirname(paths["embeddings_jsonl"]), exist_ok=True)
    with open(paths["embeddings_jsonl"], "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Embed] Saved {len(results)} embeddings to {paths['embeddings_jsonl']}")

    dims = set(len(r["embedding"]) for r in results)
    print(f"[Embed] Dimension check: {dims}")
    assert len(dims) == 1, "ERROR: Inconsistent embedding dimensions!"
    expected = emb_cfg.get("dimensions")
    if expected:
        assert dims == {expected}, f"ERROR: expected {expected} dims, got {dims}"
    print("[Embed] All embeddings have consistent dimensions ✓")


if __name__ == "__main__":
    run_embedding()
