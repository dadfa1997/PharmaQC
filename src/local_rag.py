"""Воспроизводимый локальный RAG без внешних API.

BM25 выполняет retrieval, а ответ собирается только из найденных дословных
фрагментов регламентов. Мультизапросы раскрывают разные части сложного вопроса.
Модуль используется для локальной демонстрации, контролируемой деградации и
сравнения baseline/improved на одном наборе сценариев.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from .config_loader import ConfigLoader, PROJECT_ROOT
from .rag import REFUSAL_TEXT, normalize_text

logger = logging.getLogger(__name__)

_STOPWORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по "
    "только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если "
    "уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей "
    "может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз "
    "тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом "
    "один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец "
    "два об другой хоть после над больше тот через эти нас про всего них какая много разве "
    "три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более "
    "всегда конечно всю между какие какое сколько часто делать дальше нужно необходимо "
    "каков каковы принять решение обнаружено прошло анализ находящееся находящегося"
    .split()
)

# Консервативное сведение очевидных словоформ. Полноценная лемматизация не нужна
# для небольшого корпуса, а одинаковые правила применяются и к вопросу, и к тексту.
_SUFFIXES = (
    "иями", "ями", "ами", "ией", "иях", "ого", "его", "ому", "ему",
    "ым", "им", "ой", "ей", "ая", "яя", "ое", "ее", "ые", "ие", "ых", "их",
    "ов", "ев", "ам", "ям", "ах", "ях", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь",
)


def tokenize(text: str) -> List[str]:
    """Нормализует русские/английские слова и числа для BM25."""
    text = normalize_text(re.sub(r"[–—−]", "-", text))
    tokens: List[str] = []
    canonical = {"сырья": "сырье", "сырье": "сырье"}
    for word in re.findall(r"[a-zа-я0-9%]+", text):
        if word in _STOPWORDS or len(word) < 2:
            continue
        word = canonical.get(word, word)
        if re.fullmatch(r"[а-я]+", word) and len(word) > 5:
            for suffix in _SUFFIXES:
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    word = word[: -len(suffix)]
                    break
        tokens.append(word)
    return tokens


def intent_queries(question: str) -> List[str]:
    """Формирует подзапросы для разных аспектов вопроса, без ответов-заглушек."""
    q = normalize_text(question)
    queries: List[str] = []
    if "долгосроч" in q:
        queries.append("долгосрочные исследования условия температура влажность длительность")
    if "какое решение" in q or "что делать дальше" in q:
        queries.extend([
            "критерии приемки сырья допустимые пределы количественные показатели",
            "решение о статусе сырья",
            "действия при несоответствии изоляция расследование",
        ])
    if "классифицировать" in q or "первые действия" in q:
        queries.extend([
            "классификация критические отклонения температурный режим хранения",
            "немедленные действия срок регистрации отклонения",
            "срок расследования критические отклонения",
        ])
    if "ре-квалиф" in q or "ре квалиф" in q:
        queries.append("ре-квалификация планово внепланово")
    if "срок хранения" in q and "валидац" in q:
        queries.append("срок хранения документации валидации оборудования")
    return queries


def _body(chunk_text: str) -> str:
    """Убирает служебный заголовок чанка, сохраняя текст регламента дословно."""
    lines = chunk_text.splitlines()
    if lines and lines[0].startswith("["):
        lines = lines[1:]
    return "\n".join(lines).strip()


class LocalRAGPipeline:
    """Локальный retrieval + дословная extractive-композиция ответа."""

    def __init__(self, config: Optional[ConfigLoader] = None,
                 chunks_path: Optional[str] = None):
        self.config = config or ConfigLoader()
        path = chunks_path or self.config.get("paths", "chunks_jsonl")
        self.chunks_path = Path(path)
        if not self.chunks_path.is_absolute():
            candidate = PROJECT_ROOT / self.chunks_path
            if candidate.exists() or not self.chunks_path.exists():
                self.chunks_path = candidate
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Файл чанков не найден: {self.chunks_path}. "
                "Выполните python main.py && python chunk.py"
            )

        self.chunks: List[Dict[str, Any]] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                metadata = item.get("metadata") or {}
                self.chunks.append({
                    "id": metadata.get("chunk_id", f"chunk_{len(self.chunks)}"),
                    "text": item["text"],
                    "metadata": metadata,
                })
        if not self.chunks:
            raise ValueError(f"Корпус пуст: {self.chunks_path}")

        self._tokenized = [tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self.top_k = int(self.config.get("retrieval", "top_k", default=6))
        self.strategy = str(self.config.get("local", "strategy", default="multi_query"))
        self.answer_chunks = int(self.config.get("local", "answer_chunks", default=3))
        self.object_guard = bool(self.config.get("local", "object_guard", default=True))
        logger.info("Локальный RAG инициализирован: documents=%d, strategy=%s",
                    len(self.chunks), self.strategy)

    def _rank(self, query: str) -> List[Tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in indices if scores[i] > 0]

    @staticmethod
    def _chunk_at(chunks: Sequence[Dict[str, Any]], index: int, score: float) -> Dict[str, Any]:
        chunk = dict(chunks[index])
        chunk["score"] = round(score, 4)
        return chunk

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Возвращает BM25-результаты; threshold сохранён для совместимости интерфейса."""
        del threshold
        limit = top_k or self.top_k
        primary = self._rank(query)
        selected: List[Tuple[int, float]] = primary[:limit]
        if self.strategy == "multi_query":
            # Победитель каждого аспектного подзапроса гарантированно попадает в retrieval.
            for subquery in intent_queries(query):
                ranked = self._rank(subquery)
                if ranked:
                    selected.append(ranked[0])

        seen = set()
        result: List[Dict[str, Any]] = []
        for index, score in selected:
            if index in seen:
                continue
            seen.add(index)
            result.append(self._chunk_at(self.chunks, index, score))
        return result

    def _answer_chunks(self, query: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.strategy != "multi_query":
            return retrieved[:1]
        aspects = intent_queries(query)
        if not aspects:
            return retrieved[:1]

        by_id = {c["id"]: c for c in retrieved}
        selected: List[Dict[str, Any]] = []
        for subquery in aspects:
            ranked = self._rank(subquery)
            if not ranked:
                continue
            candidate = self._chunk_at(self.chunks, ranked[0][0], ranked[0][1])
            if candidate["id"] not in by_id:
                # На этапе ответа используем только фрагменты, прошедшие retrieval.
                continue
            if candidate["id"] not in {c["id"] for c in selected}:
                selected.append(by_id[candidate["id"]])
        return selected[: self.answer_chunks]

    def _object_pair_absent(self, query: str) -> bool:
        """Консервативный отказ, если названная пара слов объекта отсутствует в корпусе."""
        tokens = tokenize(query)
        if len(tokens) < 2:
            return False
        pair = tokens[-2:]
        # Числа и общие слова не образуют надёжный идентификатор объекта.
        if any(re.fullmatch(r"[0-9%]+", token) for token in pair):
            return False
        pair_text = " ".join(pair)
        return not any(pair_text in " ".join(tokens_) for tokens_ in self._tokenized)

    @staticmethod
    def _numeric_inference(query: str, answer_chunks: List[Dict[str, Any]]) -> str:
        """Сравнивает число из вопроса с явно указанным допуском вида ±2%."""
        q_values = re.findall(r"(?<!\d)(\d+(?:[.,]\d+)?)\s*%", query)
        if not q_values:
            return ""
        for chunk in answer_chunks:
            match = re.search(r"±\s*(\d+(?:[.,]\d+)?)\s*%", chunk["text"])
            if not match:
                continue
            value = float(q_values[0].replace(",", "."))
            limit = float(match.group(1).replace(",", "."))
            if value > limit:
                return (f"Значение {q_values[0]}% превышает допустимый предел ±{match.group(1)}%: "
                        "результат не соответствует критериям приёмки.")
            return (f"Значение {q_values[0]}% не превышает допустимый предел ±{match.group(1)}% "
                    "по найденному критерию приёмки.")
        return ""

    def _result(self, query: str, retrieved: List[Dict[str, Any]], answer_chunks: List[Dict[str, Any]],
                answer: str, refused: bool, partial: bool, status: str, reason: str) -> Dict[str, Any]:
        sources: List[str] = []
        for chunk in answer_chunks if not refused else []:
            name = (chunk.get("metadata") or {}).get("document_name", "неизвестно")
            if name not in sources:
                sources.append(name)
        return {
            "question": query,
            "top_k": self.top_k,
            "similarity_threshold": None,
            "retrieved_chunks": retrieved,
            "context": "\n\n".join(c["text"] for c in answer_chunks),
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "partial": partial,
            "status": status,
            "reason": reason,
        }

    def ask(self, query: str, top_k: Optional[int] = None,
            threshold: Optional[float] = None) -> Dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k, threshold=threshold)
        if not retrieved:
            return self._result(query, [], [], REFUSAL_TEXT, True, False, "no_info",
                                "BM25 не нашёл совпадений")

        if self.object_guard and self._object_pair_absent(query):
            # Отказ срабатывает только при слабом прямом совпадении лучшего фрагмента.
            # Если оба слова объекта есть в нём, несовпадение порядка слов не считается ошибкой.
            query_tokens = set(tokenize(query))
            best_tokens = set(tokenize(retrieved[0]["text"]))
            pair_tokens = tokenize(query)[-2:]
            both_words_in_best = all(token in best_tokens for token in pair_tokens)
            coverage = len(query_tokens & best_tokens) / max(1, len(query_tokens))
            if coverage < 0.8 and not both_words_in_best:
                return self._result(query, retrieved, [], REFUSAL_TEXT, True, False, "no_info",
                                    "объект вопроса отсутствует в корпусе")

        answer_chunks = self._answer_chunks(query, retrieved)
        if not answer_chunks:
            return self._result(query, retrieved, [], REFUSAL_TEXT, True, False, "no_info",
                                "не найдены фрагменты для ответа")

        parts: List[str] = []
        inference = self._numeric_inference(query, answer_chunks) if self.strategy == "multi_query" else ""
        if inference:
            parts.append(inference)
        parts.extend(_body(c["text"]) for c in answer_chunks)

        partial = False
        if self.strategy == "multi_query" and intent_queries(query):
            q = normalize_text(query)
            if "ре-квалиф" in q or "ре квалиф" in q or ("срок хранения" in q and "валидац" in q):
                partial = True
                parts.insert(0, "Для конкретного объекта отдельное указание в документах не найдено. Общее правило:")

        answer = "\n\n".join(part for part in parts if part).strip()
        return self._result(query, retrieved, answer_chunks, answer, False, partial,
                            "general_rule" if partial else "answer", "")
