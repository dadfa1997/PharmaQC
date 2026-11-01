# Протокол проверки

Дата: 1 ноября 2026. Основной движок: local.

Проверены две среды:

- основная: Python 3.12, Linux, Uvicorn;
- независимое воспроизведение: Windows, PowerShell 5.1, Docker Desktop 4.97.0, Docker Engine 29.9.2, контейнер Python 3.11.

## 1. Автоматические тесты

Команда:

```bash
python -m pytest -q -p no:cacheprovider
```

Фактический результат: **66 passed**. Сеть, Groq API, HuggingFace и Docker для тестов не использовались.

Покрытие:

- подготовка и структурный чанкинг без потери текста;
- конфигурация и пресеты;
- BM25 retrieval, мультизапросы, отказы, partial и числовые допуски;
- Chroma/Groq pipeline на заглушках: цитаты, JSON, retry/backoff, ошибки;
- prompt injection и удаление служебных тегов;
- API: штатный ответ, health, валидация, 500/502/503 без утечки деталей;
- правила вычисления метрик.

## 2. Общие HTTP-сценарии

Запуск:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
python scripts/smoke_test.py
```

| Сценарий | Ожидаемый результат | Фактический результат | Статус |
|---|---|---|---|
| `GET /health` | HTTP 200, `ready=true` | HTTP 200, `ready=true` | PASS |
| Штатный вопрос о карантине | HTTP 200, ответ и источник | HTTP 200, `refused=false`, источник «Контроль качества сырья» | PASS |
| Вопрос без ответа в базе | HTTP 200, контролируемый отказ | HTTP 200, `refused=true`, `sources=[]` | PASS |
| Пустой вопрос | HTTP 422 | HTTP 422 | PASS |

Машиночитаемый протокол: `data/eval/http_smoke_results.json`.

## 3. Обязательные сценарии RAG

Одинаковый набор: `data/test_questions.json`. Три прогона для каждого пресета.

| ID | Тип | Что проверяется | Baseline | Improved |
|---|---|---|:---:|:---:|
| Q1 | exact | Прямой факт и источник | PASS | PASS |
| Q2 | exact | Условия и длительность исследования | PASS | PASS |
| Q3 | exact | Срок расследования | PASS | PASS |
| Q4 | exact | Перечень проверок IQ | PASS | PASS |
| Q5 | multi_hop | Допуск, решение и дальнейшие действия | FAIL | PASS |
| Q6 | partial | Оговорка об общем правиле | FAIL | PASS |
| Q7 | no_answer | Отказ без выдуманной температуры | FAIL | PASS |
| Q8 | exact | Три метода анализа причин | PASS | PASS |
| Q9 | multi_hop | Класс, действия, регистрация, расследование | FAIL | PASS |
| Q10 | partial | Общий срок хранения с оговоркой | FAIL | PASS |

Итог по каждому прогону: baseline **5/10**, improved **10/10**. Детальные ответы находятся в `data/eval/metrics_*.json`.

## 4. Сбои внешних компонентов

Проверены автотестами на управляемых заглушках, без реальных сетевых сбоев:

- 429 → повтор с задержкой → успех;
- исчерпание 429/503 → `LLMUnavailable` → HTTP 503;
- 401 → без повторов;
- пустой или невалидный JSON → повтор → `LLMBadResponse` / HTTP 502;
- непредвиденная ошибка → HTTP 500 без раскрытия внутреннего текста.

## 5. Развёртывание

### Uvicorn

Локальный процесс Uvicorn успешно инициализировал 61 чанк и обслужил HTTP-запросы. Протокол: `docs/local_run_transcript.txt`.

### Независимая проверка Docker на Windows

Команды:

```powershell
docker compose up --build -d
docker compose ps
curl.exe http://127.0.0.1:8000/health
python scripts/smoke_test.py
pytest -p no:cacheprovider
```

| Проверка | Фактический результат | Статус |
|---|---|:---:|
| Сборка `pharmaqc-pharmaqc-api` | 14/14 шагов, образ собран | PASS |
| Контейнер `pharmaqc-api` | `Up ... (healthy)`, порт 8000 проброшен | PASS |
| Docker Engine | Linux/amd64, версия 29.9.2 | PASS |
| `GET /health` | HTTP 200, `ready=true` | PASS |
| HTTP-сценарии | health, normal, no_answer, validation — 4/4 | PASS |
| Автотесты | 66 passed in 1.19s | PASS |
| Baseline, 3 прогона | accuracy 0.50 во всех прогонах | воспроизведено |
| Improved, 3 прогона | accuracy 1.00, Hit@k 1.000, 0 галлюцинаций | цели достигнуты |

Источники: `docs/docker_windows_transcript.txt`, `data/eval/summary_baseline_20261101_201240.json`, `data/eval/summary_improved_20261101_201241.json`, `data/eval/http_smoke_results.json`. Максимальный p95 improved в независимой серии — 1.7 мс, что значительно ниже цели 10 000 мс.

## 6. Остаточные ограничения проверки

- 10 вопросов недостаточны для статистически устойчивой оценки производственного качества.
- Набор содержит только один no_answer; нужны дополнительные негативные и adversarial-сценарии.
- Проверка required_facts автоматическая и не заменяет независимую экспертизу смысла.
- Реальные latency и качество Groq не измерялись из-за отсутствия ключа.
