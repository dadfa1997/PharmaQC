import json

import pytest

from src.config_loader import ConfigLoader
from src.local_rag import LocalRAGPipeline, intent_queries, tokenize
from src.rag import REFUSAL_TEXT


@pytest.fixture(scope="module")
def questions():
    with open("data/test_questions.json", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def improved():
    return LocalRAGPipeline(ConfigLoader("improved"))


def test_tokenizer_normalizes_domain_word_forms():
    assert tokenize("сырьё и сырья") == ["сырье", "сырье"]
    assert tokenize("таблеточного пресса")[-2:] == tokenize("таблеточных прессов")


def test_improved_answers_exact_question_with_source(improved, questions):
    result = improved.ask(questions[0]["question"])
    assert not result["refused"]
    assert "жёлтая этикетка" in result["answer"]
    assert result["sources"] == ["Контроль качества сырья"]


def test_improved_refuses_unknown_object(improved, questions):
    result = improved.ask(questions[6]["question"])
    assert result["refused"] is True
    assert result["answer"] == REFUSAL_TEXT
    assert result["sources"] == []


def test_improved_marks_general_rule_as_partial(improved, questions):
    for index in (5, 9):
        result = improved.ask(questions[index]["question"])
        assert result["partial"] is True
        assert result["status"] == "general_rule"
        assert "отдельное указание" in result["answer"]


def test_multi_hop_combines_required_fragments(improved, questions):
    result = improved.ask(questions[8]["question"])
    for fact in ("Критические отклонения", "изолировать", "24 часов", "30 календарных дней"):
        assert fact in result["answer"]
    assert result["sources"] == ["Управление отклонениями"]


def test_numeric_rule_compares_query_with_document_limit(improved, questions):
    result = improved.ask(questions[4]["question"])
    assert "3% превышает допустимый предел ±2%" in result["answer"]
    assert "СОП-ОКК-015" in result["answer"]


def test_intent_queries_cover_complex_questions(questions):
    assert len(intent_queries(questions[4]["question"])) == 3
    assert len(intent_queries(questions[8]["question"])) == 3
    assert intent_queries(questions[0]["question"]) == []


def test_improved_beats_single_query_baseline_on_same_set(questions):
    baseline = LocalRAGPipeline(ConfigLoader("baseline"))
    improved = LocalRAGPipeline(ConfigLoader("improved"))

    def correct(pipeline, question):
        result = pipeline.ask(question["question"])
        if question["type"] == "no_answer":
            return result["refused"]
        return not result["refused"]

    baseline_score = sum(correct(baseline, q) for q in questions)
    improved_score = sum(correct(improved, q) for q in questions)
    assert improved_score > baseline_score
