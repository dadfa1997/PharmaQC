import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402

Q = {q["id"]: q for q in json.loads((ROOT / "data" / "test_questions.json").read_text(encoding="utf-8"))}


def res(answer, refused=False):
    return {"answer": answer, "refused": refused}


def test_exact_correct_and_false_refusal():
    assert ev.judge(Q[1], res("Жёлтая этикетка «КАРАНТИН» с датой поступления и номером серии."))["correct"]
    assert ev.judge(Q[1], res("К сожалению…", True))["false_refusal"]


def test_no_answer_must_refuse():
    assert ev.judge(Q[7], res("К сожалению…", True))["correct"]
    assert ev.judge(Q[7], res("Не выше 25°C"))["hallucination"]


def test_partial_needs_caveat_or_cautious_refusal():
    assert ev.judge(Q[6], res("Для пресса отдельно не указано; общее правило — каждые 2–3 года."))["correct"]
    assert ev.judge(Q[6], res("Каждые 2–3 года."))["hallucination"]
    assert ev.judge(Q[6], res("К сожалению…", True))["cautious_refusal"]


def test_min_groups_allows_partial_fact_coverage():
    ans = "Критическое отклонение; изолировать продукцию; регистрация в течение 24 часов."
    assert ev.judge(Q[9], res(ans))["correct"]          # 3 из 4 групп при min_groups=3


def test_percentile_nearest_rank():
    assert ev.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert ev.percentile([], 95) == 0.0


def test_questions_file_is_consistent():
    for q in Q.values():
        assert q["type"] in {"exact", "multi_hop", "no_answer", "partial"}
        if q["type"] != "no_answer":
            assert q.get("required_facts"), q["id"]
        if q["type"] == "partial":
            assert q.get("caveat_markers"), q["id"]
