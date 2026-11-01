"""Этап 2: разбиение подготовленных документов на чанки -> data/chunks/chunks.jsonl."""
import json
import os

from src.chunker import TextChunker
from src.config_loader import ConfigLoader


def run_chunking(config: ConfigLoader = None):
    config = config or ConfigLoader()
    cfg = config.section("chunking")
    paths = config.section("paths")

    chunker = TextChunker(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        min_chunk_size=cfg.get("min_chunk_size", 50),
    )

    documents = []
    with open(paths["prepared_jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    print(f"[Chunker] Loaded {len(documents)} documents from {paths['prepared_jsonl']}")

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk_document(doc))
    print(f"[Chunker] Produced {len(all_chunks)} chunks")

    os.makedirs(os.path.dirname(paths["chunks_jsonl"]), exist_ok=True)
    with open(paths["chunks_jsonl"], "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[Chunker] Saved to {paths['chunks_jsonl']}")

    sizes = [c["metadata"]["chunk_size"] for c in all_chunks]
    if sizes:
        print(f"[Chunker] Chunk sizes - min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    run_chunking()
