FROM python:3.11-slim

WORKDIR /app

# curl нужен для healthcheck в docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# По умолчанию используется локальный BM25 RAG: внешние модели и ключи не нужны.
# Для PHARMAQC_ENGINE=groq модель эмбеддингов скачивается при первом запуске.
ENV PHARMAQC_ENGINE=local \
    PYTHONUNBUFFERED=1

COPY . .

EXPOSE 8000

# Для groq-режима индекс data/vector_store/chroma_db собирается заранее и монтируется через volume
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
