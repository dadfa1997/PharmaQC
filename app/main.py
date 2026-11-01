"""FastAPI-приложение PharmaQC."""
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from src.config_loader import ConfigLoader  # noqa: E402
from src.errors import LLMBadResponse, LLMUnavailable  # noqa: E402
from src.service import PharmaQCService, get_service  # noqa: E402

_config = ConfigLoader()
MAX_QUESTION_LENGTH = _config.get("api", "max_question_length", default=1000)
VERSION = str(_config.get("version", default="2.1.0"))

logging.basicConfig(
    level=getattr(logging, str(_config.get("logging", "level", default="INFO")).upper(), logging.INFO),
    format=_config.get("logging", "format", default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH,
                          description=f"Текст вопроса (1-{MAX_QUESTION_LENGTH} символов)")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Вопрос не должен быть пустым")
        return v

    model_config = {"json_schema_extra": {"example": {"question": "Как маркируется сырьё, находящееся в карантине?"}}}


class ChunkInfo(BaseModel):
    id: str
    score: float
    document: str
    text_preview: str


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Текст ответа или стандартная фраза отказа")
    sources: List[str] = Field(default_factory=list, description="Документы, на которые опирается ответ")
    chunks: List[ChunkInfo] = Field(default_factory=list, description="Найденные фрагменты")
    latency_ms: float = Field(..., description="Время обработки, мс")
    refused: bool = Field(..., description="True, если система отказалась отвечать")
    partial: bool = Field(False, description="True, если в документах только общее правило, а не прямой ответ")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "PharmaQC RAG API"
    version: str = VERSION
    ready: bool = False


service: Optional[PharmaQCService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("Запуск PharmaQC RAG API...")
    try:
        service = get_service()
    except Exception:
        logger.exception("Не удалось инициализировать сервис")
        raise
    logger.info("Сервис инициализирован")
    yield
    logger.info("Завершение работы PharmaQC RAG API")


app = FastAPI(
    title="PharmaQC RAG API",
    description="RAG-ассистент для отдела контроля качества фармацевтического производства",
    version=VERSION,
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()] \
    or list(_config.get("api", "cors_origins", default=[]) or [])
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.exception_handler(LLMUnavailable)
async def llm_unavailable_handler(request: Request, exc: LLMUnavailable):
    logger.error("LLM недоступна: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Сервис генерации ответов временно недоступен. Повторите запрос позже."})


@app.exception_handler(LLMBadResponse)
async def llm_bad_response_handler(request: Request, exc: LLMBadResponse):
    logger.error("Некорректный ответ LLM: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "Не удалось получить корректный ответ от модели. Повторите запрос."})


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Процесс жив; `ready` показывает, инициализирован ли RAG-сервис."""
    return HealthResponse(ready=service is not None)


@app.post("/api/v1/ask", response_model=QueryResponse, tags=["RAG"])
async def ask_question(request: QueryRequest):
    """Задать вопрос по регламентам. Возвращает ответ, источники и найденные фрагменты."""
    if service is None:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    try:
        # Синхронный вызов (модель, Chroma, сеть) выполняем в пуле потоков, чтобы не блокировать event loop
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(service.ask, request.question)
    except (LLMUnavailable, LLMBadResponse):
        raise
    except Exception:
        logger.exception("Непредвиденная ошибка при обработке вопроса")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    return QueryResponse(**result)


@app.get("/", tags=["Root"])
async def root():
    return {"message": "PharmaQC RAG API", "docs": "/docs", "health": "/health", "ask": "/api/v1/ask"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
