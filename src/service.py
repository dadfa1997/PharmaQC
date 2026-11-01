"""Сервисный слой: чистая функция ask() без привязки к HTTP.

Тяжёлые зависимости (torch, chromadb, groq) импортируются лениво — модуль можно
импортировать и тестировать без них.
"""
import logging
import os
import time
from typing import Any, Dict, Optional

from .config_loader import ConfigLoader
from .rag import RAGPipeline

logger = logging.getLogger(__name__)


class PharmaQCService:
    """Собирает RAG-пайплайн по конфигу и отвечает на вопросы.

    Ошибки LLM (LLMUnavailable, LLMBadResponse) не скрываются: вызывающий код сам решает,
    как их показать (HTTP 503/502, повтор и т.д.).
    """

    def __init__(self, preset_name: Optional[str] = None, config: Optional[ConfigLoader] = None,
                 rag: Optional[RAGPipeline] = None):
        self.config = config or ConfigLoader(preset_name)
        self.preset_name = preset_name
        if rag is not None:          # внедрение готового пайплайна (тесты, нестандартная сборка)
            self.rag = rag
            return
        self.rag = self._build_rag()

    def _build_rag(self) -> RAGPipeline:
        engine = os.getenv("PHARMAQC_ENGINE") or self.config.get("runtime", "engine", default="local")
        if engine == "local":
            from .local_rag import LocalRAGPipeline

            return LocalRAGPipeline(config=self.config)
        if engine != "groq":
            raise ValueError(f"Неизвестный PHARMAQC_ENGINE: {engine}. Допустимо: local, groq")

        import chromadb
        from chromadb.config import Settings
        from groq import Groq
        from sentence_transformers import SentenceTransformer

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY не найден в переменных окружения")

        model_name = self.config.get("embedding", "model_name")
        logger.info("Загрузка модели эмбеддингов: %s", model_name)
        embedder = SentenceTransformer(model_name, device=self.config.get("embedding", "device", default="cpu"))

        chroma_path = self.config.get("vector_store", "persist_directory")
        collection_name = self.config.get("vector_store", "collection_name")
        logger.info("Подключение к ChromaDB: %s / %s", chroma_path, collection_name)
        client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:  # коллекции нет — индекс не собран
            raise RuntimeError(
                f"Коллекция '{collection_name}' не найдена в {chroma_path}. "
                "Соберите индекс: python main.py && python chunk.py && python embed.py && python index.py"
            ) from exc

        return RAGPipeline(collection=collection, embedder_model=embedder, llm_client=Groq(api_key=api_key),
                           config=self.config)

    def ask(self, question: str) -> Dict[str, Any]:
        started = time.perf_counter()
        result = self.rag.ask(question)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        chunks = [
            {
                "id": c["id"],
                "score": c["score"],
                "document": c["metadata"].get("document_name", "неизвестно"),
                "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            }
            for c in result["retrieved_chunks"]
        ]
        # Текст вопроса в лог не пишем: только размеры и идентификаторы.
        logger.info("Вопрос обработан: len=%d, chunks=%d, refused=%s, status=%s, %.0f мс",
                    len(question), len(chunks), result["refused"], result["status"], latency_ms)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": chunks,
            "latency_ms": latency_ms,
            "refused": result["refused"],
            "partial": result["partial"],
        }


_service_instance: Optional[PharmaQCService] = None


def get_service(preset_name: Optional[str] = None) -> PharmaQCService:
    """Singleton-сервис (создаётся при первом обращении)."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PharmaQCService(preset_name)
    return _service_instance


def ask(question: str) -> Dict[str, Any]:
    """Чистая функция для использования из других модулей (например, как инструмент агента)."""
    return get_service().ask(question)
