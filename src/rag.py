"""RAG: retrieval -> контекст -> LLM -> проверка опоры ответа в контексте."""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import ConfigLoader
from .errors import LLMBadResponse, LLMUnavailable

logger = logging.getLogger(__name__)

REFUSAL_TEXT = "К сожалению, в доступных документах нет информации для ответа на этот вопрос."
VALID_STATUSES = {"answer", "general_rule", "no_info"}
GROUNDING_MIN_RATIO = 0.85   # доля слов цитаты, которые должны найтись в контексте
_TAG_RE = re.compile(r"</?\s*(context|question)\s*>", re.IGNORECASE)


# ── вспомогательные функции ───────────────────────────────────────

def normalize_text(text: str) -> str:
    """Нижний регистр, единые дефисы/кавычки/пробелы, ё→е — для сравнения цитат с контекстом."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)   # все виды дефисов и тире
    text = re.sub(r"[«»“”„\"']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[\w%°±+.,/-]+", normalize_text(text))


def quote_grounding(quote: str, text: str) -> float:
    """Доля слов цитаты, присутствующих в тексте (1.0 при точном вхождении)."""
    q = normalize_text(quote)
    if not q:
        return 0.0
    t = normalize_text(text)
    if q in t:
        return 1.0
    q_tokens = _tokens(quote)
    if not q_tokens:
        return 0.0
    t_tokens = set(_tokens(text))
    return sum(1 for tok in q_tokens if tok in t_tokens) / len(q_tokens)


def split_quote(quote: str) -> List[str]:
    parts = re.split(r"\s*(?:\||…|\.\.\.)\s*", quote or "")
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def sanitize_for_prompt(text: str) -> str:
    """Убирает из пользовательского текста и контекста служебные теги-разделители."""
    return _TAG_RE.sub("", text)


def parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    """Достаёт JSON-объект из ответа модели (допускает обёртку в ```json и текст вокруг)."""
    if not content:
        return None
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def classify_llm_error(exc: Exception) -> str:
    """rate_limit | transient | fatal — без жёсткой привязки к классам SDK."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if status == 429 or "ratelimit" in name or "rate_limit" in msg:
        return "rate_limit"
    if (isinstance(status, int) and status >= 500) or "timeout" in name or "connection" in name:
        return "transient"
    return "fatal"


# ── пайплайн ──────────────────────────────────────────────────────

class RAGPipeline:
    """End-to-end RAG: retrieval → контекст → LLM → проверка опоры в контексте."""

    CONTEXT_TEMPLATE = "[Фрагмент {index}] Документ: {source}\n{text}"

    def __init__(self, collection, embedder_model, llm_client=None,
                 preset_name: Optional[str] = None, config: Optional[ConfigLoader] = None):
        self.collection = collection
        self.embedder = embedder_model
        self.llm_client = llm_client
        self.config = config or ConfigLoader(preset_name)

        c = self.config
        self.query_prefix = c.get("embedding", "query_prefix", default="query: ")
        self.top_k = c.get("retrieval", "top_k", default=5)
        self.threshold = c.get("retrieval", "similarity_threshold", default=0.75)
        self.llm_model = c.get("llm", "model")
        self.max_tokens = c.get("llm", "max_tokens", default=2000)
        self.temperature = c.get("llm", "temperature", default=0.0)
        self.timeout = c.get("llm", "timeout", default=30)
        self.max_retries = c.get("llm", "max_retries", default=3)
        self.retry_delay = c.get("llm", "retry_delay", default=5)
        self.reasoning_effort = c.get("llm", "reasoning_effort")
        self.json_mode = c.get("llm", "json_mode", default=True)
        self.system_prompt = c.get("prompt", "system_prompt", default="")
        self._sleep = time.sleep   # подменяется в тестах

    # ── retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold

        embedding = self.embedder.encode([self.query_prefix + query], normalize_embeddings=True)[0].tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            score = round(1 - results["distances"][0][i], 4)
            if score < threshold:
                continue
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": score,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })
        return chunks

    # ── контекст ──────────────────────────────────────────────────

    @staticmethod
    def unique_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, unique = set(), []
        for c in chunks:
            key = normalize_text(c["text"])
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(c)
        return unique

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, c in enumerate(self.unique_chunks(chunks), 1):
            meta = c.get("metadata", {})
            parts.append(self.CONTEXT_TEMPLATE.format(
                index=i,
                source=meta.get("document_name", meta.get("source", "неизвестно")),
                text=sanitize_for_prompt(c["text"]),
            ))
        return "\n\n".join(parts)

    # ── LLM ───────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]):
        kwargs: Dict[str, Any] = dict(
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return self.llm_client.chat.completions.create(**kwargs)

    def generate(self, query: str, context: str) -> Dict[str, Any]:
        """Возвращает {"status", "answer", "quote"}. Бросает LLMUnavailable / LLMBadResponse."""
        if self.llm_client is None:
            raise LLMUnavailable("LLM-клиент не подключён")

        user = (f"<context>\n{context}\n</context>\n\n"
                f"<question>\n{sanitize_for_prompt(query)}\n</question>")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

        last_problem = "нет попыток"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._call_llm(messages)
            except Exception as exc:  # noqa: BLE001 — классифицируем ниже
                kind = classify_llm_error(exc)
                logger.warning("Ошибка LLM (%s, попытка %d/%d): %s",
                               kind, attempt, self.max_retries, type(exc).__name__)
                if kind == "fatal":
                    raise LLMUnavailable(f"Ошибка LLM: {type(exc).__name__}") from exc
                last_problem = f"{kind}: {type(exc).__name__}"
                if attempt < self.max_retries:
                    delay = self.retry_delay * attempt if kind == "rate_limit" else self.retry_delay
                    self._sleep(delay)
                continue

            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            finish = getattr(choice, "finish_reason", None)
            if not content:
                # Пустой ответ — не отказ. У рассуждающих моделей так бывает, если reasoning съел max_tokens.
                logger.warning("Пустой ответ LLM (finish_reason=%s, попытка %d/%d)",
                               finish, attempt, self.max_retries)
                last_problem = f"пустой ответ (finish_reason={finish})"
                continue

            data = parse_llm_json(content)
            status = str(data.get("status", "")).strip().lower() if data else ""
            if data is None or status not in VALID_STATUSES:
                logger.warning("Ответ LLM не соответствует формату (попытка %d/%d)", attempt, self.max_retries)
                last_problem = "ответ не соответствует формату JSON"
                continue

            return {
                "status": status,
                "answer": str(data.get("answer", "")).strip(),
                "quote": str(data.get("quote", "")).strip(),
            }

        if "rate_limit" in last_problem or "transient" in last_problem:
            raise LLMUnavailable(f"LLM недоступна после {self.max_retries} попыток ({last_problem})")
        raise LLMBadResponse(f"LLM не дала корректный ответ после {self.max_retries} попыток ({last_problem})")

    # ── проверка опоры и источники ────────────────────────────────

    def check_grounding(self, quote: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Каждая часть цитаты должна найтись в контексте. Возвращает (ok, чанки-опоры)."""
        parts = split_quote(quote)
        if not parts:
            return False, []
        supporting: List[Dict[str, Any]] = []
        for part in parts:
            matched = [c for c in chunks if quote_grounding(part, c["text"]) >= GROUNDING_MIN_RATIO]
            if not matched:
                return False, []
            for c in matched:
                if c not in supporting:
                    supporting.append(c)
        return True, supporting

    @staticmethod
    def sources_of(chunks: List[Dict[str, Any]]) -> List[str]:
        names: List[str] = []
        for c in chunks:
            name = (c.get("metadata") or {}).get("document_name", "неизвестно")
            if name not in names:
                names.append(name)
        return names

    # ── полный цикл ───────────────────────────────────────────────

    def _result(self, query, chunks, context, answer, sources, refused, partial, status, reason, top_k, threshold):
        return {
            "question": query,
            "top_k": top_k,
            "similarity_threshold": threshold,
            "retrieved_chunks": chunks,
            "context": context,
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "partial": partial,
            "status": status,
            "reason": reason,
        }

    def ask(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> Dict[str, Any]:
        top_k = top_k or self.top_k
        threshold = self.threshold if threshold is None else threshold

        chunks = self.retrieve(query, top_k, threshold)
        if not chunks:
            return self._result(query, [], "", REFUSAL_TEXT, [], True, False, "no_info",
                                "нет чанков выше порога", top_k, threshold)

        unique = self.unique_chunks(chunks)
        context = self.build_context(unique)
        llm = self.generate(query, context)

        if llm["status"] == "no_info":
            return self._result(query, chunks, context, REFUSAL_TEXT, [], True, False, "no_info",
                                "модель: в контексте нет ответа", top_k, threshold)

        ok, supporting = self.check_grounding(llm["quote"], unique)
        if not ok or not llm["answer"]:
            logger.info("Ответ отклонён: цитата не найдена в контексте")
            return self._result(query, chunks, context, REFUSAL_TEXT, [], True, False, "ungrounded",
                                "цитата не найдена в контексте", top_k, threshold)

        return self._result(query, chunks, context, llm["answer"], self.sources_of(supporting),
                            False, llm["status"] == "general_rule", llm["status"], "", top_k, threshold)
