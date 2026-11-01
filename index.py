"""Этап 4: индексация эмбеддингов в ChromaDB с валидацией."""
from src.config_loader import ConfigLoader
from src.indexer import VectorStoreIndexer


def run_indexing(config: ConfigLoader = None):
    config = config or ConfigLoader()
    VectorStoreIndexer(config.indexer_config(), config_path=str(config.config_path)).run()


if __name__ == "__main__":
    run_indexing()
