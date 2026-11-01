# Сценарий демонстрации PharmaQC

Продолжительность: 3–4 минуты. Демонстрация не требует API-ключей и сети после установки зависимостей.

Готовые материалы:

- `demo/PharmaQC_демонстрация.mp4` — короткая видеодемонстрация (36 секунд), составленная из фактического вывода локального запуска: Uvicorn, health, четыре HTTP-сценария, multi-hop Q9, 66 автотестов и метрики;
- `docs/local_run_transcript.txt` — полный текстовый протокол запуска через Uvicorn;
- `docs/docker_windows_transcript.txt` — протокол независимой проверки на Windows: сборка Docker-образа, контейнер `healthy`, 4/4 HTTP PASS, 66 тестов и повторное сравнение baseline/improved.

Видео подтверждает запуск через Uvicorn. Запуск Docker отдельно подтверждён текстовым протоколом и машиночитаемыми результатами в `data/eval/`; содержимое видео не выдаётся за запись Docker-запуска.

## 1. Запуск

Вариант через Uvicorn:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Вариант через Docker, независимо проверенный на Windows:

```powershell
docker compose up --build -d
docker compose ps
```

Показать состояние контейнера `healthy` либо в логе Uvicorn строку: `Локальный RAG инициализирован: documents=61, strategy=multi_query`.

## 2. Health и штатный запрос

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как маркируется сырьё, находящееся в карантине?"}'
```

На что обратить внимание: ответ содержит жёлтую этикетку «КАРАНТИН», дату и номер серии; `sources` указывает регламент; `refused=false`.

## 3. Многошаговый запрос

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Обнаружено нарушение температурного режима хранения сырья. Как классифицировать отклонение и каковы первые действия, срок регистрации и срок расследования?"}'
```

Ожидается объединение трёх частей: критическое отклонение; изоляция и регистрация за 24 часа; расследование до 30 календарных дней.

## 4. Отказ и обработка ошибки пользователя

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Какая максимальная температура хранения инъекционных растворов?"}'

curl -i -X POST http://127.0.0.1:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"   "}'
```

Первый запрос возвращает `refused=true` и пустые источники, второй — HTTP 422. Сервис продолжает работать.

## 5. Автоматическая повторяемая демонстрация

```bash
python scripts/smoke_test.py
pytest -q -p no:cacheprovider
python scripts/evaluate.py --preset baseline --runs 3 --pause 0
python scripts/evaluate.py --preset improved --runs 3 --pause 0
```

Финальный кадр: 66 passed; baseline accuracy=0.50; improved accuracy=1.00; HTTP smoke — 4/4 PASS.

## 6. Что важно озвучить

- Local-режим отвечает дословными фрагментами и выбран для воспроизводимой демонстрации.
- Groq-режим сохранён как отдельный генеративный вариант, но его качество этими цифрами не подтверждается.
- Результат 10/10 относится только к фиксированному учебному набору.
