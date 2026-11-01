import re
from pathlib import Path

import pytest

from src.chunker import TextChunker
from src.cleaner import TextCleaner

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _doc(text, name="Тестовый регламент"):
    return {"text": text, "metadata": {"document_id": "doc_x", "document_name": name}}


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_one_line_clauses_are_not_lost():
    text = "Заголовок\nВерсия 1\n\n1. Раздел\n1.1. Первый пункт без списка.\n1.2. Второй пункт без списка.\n"
    chunks = TextChunker(400, 80, 10).chunk_document(_doc(text))
    joined = " ".join(c["text"] for c in chunks)
    assert "Первый пункт без списка." in joined
    assert "Второй пункт без списка." in joined


def test_every_chunk_has_document_and_section_prefix():
    text = "Шапка\n\n2. Приёмка сырья\n2.3. Маркировка: жёлтая этикетка «КАРАНТИН».\n"
    chunk = TextChunker(400, 80, 10).chunk_document(_doc(text, "Контроль сырья"))[0]
    assert chunk["text"].startswith("[Контроль сырья] 2. Приёмка сырья\n")
    assert chunk["metadata"]["section_number"] == "2"
    assert chunk["metadata"]["clauses"] == "2.3"


def test_ids_are_stable_and_unique():
    text = "Шапка\n\n1. А\n1.1. " + "слово " * 30 + "\n\n2. Б\n2.1. " + "текст " * 30 + "\n"
    chunks = TextChunker(100, 20, 10).chunk_document(_doc(text))
    ids = [c["metadata"]["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "doc_x_chunk0"


def test_long_clause_is_split_and_keeps_lead_line():
    items = "\n".join(f"- условие номер {i} со значением {i * 7}" for i in range(30))
    text = f"Шапка\n\n2. Виды\n2.1. Долгосрочные исследования:\n{items}\n"
    chunks = TextChunker(200, 40, 10).chunk_document(_doc(text))
    assert len(chunks) > 1
    assert all("2.1. Долгосрочные исследования:" in c["text"] or i == 0 for i, c in enumerate(chunks))
    assert all("2.1. Долгосрочные исследования" in c["text"] for c in chunks)


@pytest.mark.parametrize("path", sorted(RAW_DIR.glob("*.txt")), ids=lambda p: p.name)
def test_no_text_lost_on_real_documents(path):
    """Регрессия: структурный чанкер раньше терял однострочные пункты (~28% текста)."""
    text = TextCleaner().clean(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
    chunks = TextChunker(400, 80, 50).chunk_document(_doc(text, path.stem))
    blob = _norm(" ".join(c["text"] for c in chunks))
    started, lost = False, []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        started = started or bool(re.match(r"^\d+\.\s", line))
        if started and _norm(line) not in blob:
            lost.append(line)
    assert not lost, f"потеряны строки: {lost[:3]}"
