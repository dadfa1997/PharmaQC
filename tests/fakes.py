"""Заглушки эмбеддера, ChromaDB и Groq: тесты не требуют сети, моделей и ключей."""
import json
from types import SimpleNamespace
from typing import Any, List


class Vec(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def __init__(self):
        self.calls: List[str] = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.extend(texts)
        return [Vec([0.1, 0.2, 0.3]) for _ in texts]


class FakeCollection:
    """rows: [(id, text, document_name, score)] — в порядке убывания score."""

    def __init__(self, rows):
        self.rows = rows
        self.last_n = None

    def query(self, query_embeddings, n_results, include):
        self.last_n = n_results
        rows = self.rows[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[{"document_name": r[2], "category": "test"} for r in rows]],
            "distances": [[round(1 - r[3], 4) for r in rows]],
        }


def llm_json(status="answer", answer="Ответ.", quote=""):
    return json.dumps({"status": status, "answer": answer, "quote": quote}, ensure_ascii=False)


def make_response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                    finish_reason=finish_reason)])


class FakeLLM:
    """Каждый вызов берёт следующий элемент: строка (ответ), Exception (бросается) или готовый response."""

    def __init__(self, script: List[Any]):
        self.script = list(script)
        self.calls: List[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return make_response(item)
        return item


class HttpError(Exception):
    def __init__(self, status_code, msg="error"):
        super().__init__(msg)
        self.status_code = status_code
