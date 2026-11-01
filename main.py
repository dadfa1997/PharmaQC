"""Этап 1: подготовка данных (загрузка, очистка, нормализация) -> data/prepared/dataset.jsonl."""
from src.config_loader import ConfigLoader
from src.pipeline import RAGPrepPipeline

if __name__ == "__main__":
    RAGPrepPipeline(ConfigLoader()).run()
