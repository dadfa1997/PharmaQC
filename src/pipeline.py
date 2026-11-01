import os

from .config_loader import ConfigLoader
from .loader import DataLoader
from .cleaner import TextCleaner
from .normalizer import TextNormalizer
from .structurer import DocumentStructurer
from .exporter import DataExporter


class RAGPrepPipeline:
    """Подготовка данных: загрузка, очистка, нормализация, структурирование, экспорт в JSONL."""

    def __init__(self, config: ConfigLoader = None):
        self.config = config or ConfigLoader()
        self.paths = self.config.section("paths")
        self.pre = self.config.section("preprocessing")

        self.loader = DataLoader()
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer(normalize_unicode=self.pre.get("normalize_unicode", True))
        self.structurer = DocumentStructurer()
        self.exporter = DataExporter()

    def run(self):
        print("=" * 50)
        print("STARTING RAG DATA PREPARATION PIPELINE")
        print("=" * 50)

        raw_docs = self.loader.load(self.paths["raw_data_dir"], self.paths["documents_metadata"])
        if not raw_docs:
            print("[Pipeline] No documents found. Exiting.")
            return

        processed_docs = []
        min_len = self.pre.get("min_text_length", 10)

        for doc in raw_docs:
            text = self.cleaner.clean(doc["raw_text"])
            text = self.normalizer.normalize(text)

            if len(text) < min_len:
                print(f"[Pipeline] Skipped {doc['filename']}: too short after cleaning")
                continue

            doc["raw_text"] = text
            processed_docs.append(self.structurer.structure(doc))

        self.exporter.export_jsonl(processed_docs, self.paths["prepared_jsonl"])

        print(f"[Pipeline] Successfully processed {len(processed_docs)}/{len(raw_docs)} documents")
        print("=" * 50)
