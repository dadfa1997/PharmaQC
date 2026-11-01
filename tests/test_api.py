import pytest
from fastapi.testclient import TestClient

import app.main as main
from src.errors import LLMBadResponse, LLMUnavailable


class FakeService:
    def __init__(self, result=None, error=None):
        self.result = result or {"answer": "Ответ", "sources": ["Документ"], "chunks": [], "latency_ms": 12.5,
                                 "refused": False, "partial": False}
        self.error = error
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client():
    old = main.service
    main.service = FakeService()
    yield TestClient(main.app)
    main.service = old


def test_health_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["ready"] is True


def test_health_not_ready():
    old = main.service
    main.service = None
    try:
        assert TestClient(main.app).get("/health").json()["ready"] is False
    finally:
        main.service = old


def test_ask_ok(client):
    r = client.post("/api/v1/ask", json={"question": "  Как маркируется карантин?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Ответ" and body["sources"] == ["Документ"] and body["refused"] is False
    assert main.service.questions == ["Как маркируется карантин?"]    # пробелы по краям убраны


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {}, {"question": "а" * 1001}, {"question": 5}])
def test_validation_errors(client, payload):
    assert client.post("/api/v1/ask", json=payload).status_code == 422


def test_max_length_boundary_ok(client):
    assert client.post("/api/v1/ask", json={"question": "а" * 1000}).status_code == 200


def test_service_not_initialized():
    old = main.service
    main.service = None
    try:
        assert TestClient(main.app).post("/api/v1/ask", json={"question": "в"}).status_code == 503
    finally:
        main.service = old


def test_llm_unavailable_maps_to_503_without_details(client):
    main.service = FakeService(error=LLMUnavailable("token gsk_secret rate limit"))
    r = client.post("/api/v1/ask", json={"question": "в"})
    assert r.status_code == 503
    assert "gsk_secret" not in r.text


def test_llm_bad_response_maps_to_502(client):
    main.service = FakeService(error=LLMBadResponse("пустой ответ"))
    assert client.post("/api/v1/ask", json={"question": "в"}).status_code == 502


def test_unexpected_error_maps_to_500_without_leak():
    old = main.service
    main.service = FakeService(error=RuntimeError("internal path /secret/db"))
    try:
        r = TestClient(main.app, raise_server_exceptions=False).post("/api/v1/ask", json={"question": "в"})
        assert r.status_code == 500
        assert "/secret/db" not in r.text
    finally:
        main.service = old


def test_error_text_never_in_answer(client):
    main.service = FakeService(error=LLMUnavailable("x"))
    r = client.post("/api/v1/ask", json={"question": "в"})
    assert "answer" not in r.json()


def test_refusal_response_shape(client):
    main.service = FakeService(result={"answer": "К сожалению...", "sources": [], "chunks": [], "latency_ms": 1.0,
                                       "refused": True, "partial": False})
    body = client.post("/api/v1/ask", json={"question": "в"}).json()
    assert body["refused"] is True and body["sources"] == []
