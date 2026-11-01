"""Оценка качества RAG на data/test_questions.json. Метрики считаются кодом.

  python scripts/evaluate.py --preset improved --runs 3

Правила оценки по типам вопросов:
  exact / multi_hop — не отказ + все группы required_facts (или min_groups) найдены в ответе;
  no_answer         — верно только отказ; любой ответ = галлюцинация;
  partial           — верно: ответ с required_facts и оговоркой (caveat_markers) ИЛИ осторожный отказ;
                      уверенный ответ без оговорки = галлюцинация (перегенерализация).
hit@k считается по найденным чанкам (retrieval), а не по тексту ответа.
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.errors import LLMError  # noqa: E402
from src.rag import normalize_text  # noqa: E402

TARGET_ACCURACY = 0.8
TARGET_P95_MS = 10_000


def facts_found(answer: str, q: dict):
    groups = q.get("required_facts") or []
    if not groups:
        return True, []
    a = normalize_text(answer)
    missing = [g for g in groups if not any(normalize_text(alt) in a for alt in g)]
    need = q.get("min_groups", len(groups))
    return (len(groups) - len(missing)) >= need, missing


def hit_at_k(q: dict, chunks: list) -> bool:
    expected = q.get("expected_source")
    if not expected:
        return False
    names = {(c.get("metadata") or {}).get("document_name") for c in chunks}
    return all(part.strip() in names for part in expected.split("+"))


def judge(q: dict, res: dict) -> dict:
    refused, answer = res["refused"], res["answer"]
    t = q["type"]
    out = {"correct": False, "hallucination": False, "false_refusal": False, "cautious_refusal": False, "reason": ""}
    if t == "no_answer":
        out["correct"] = refused
        out["hallucination"] = not refused
        out["reason"] = "верный отказ" if refused else "ответил, хотя в документах ответа нет"
    elif t == "partial":
        if refused:
            out.update(correct=True, cautious_refusal=True, reason="осторожный отказ (допустимо)")
        else:
            ok, missing = facts_found(answer, q)
            caveat = any(normalize_text(m) in normalize_text(answer) for m in q.get("caveat_markers", []))
            out["correct"] = ok and caveat
            out["hallucination"] = not caveat
            out["reason"] = ("ответ с оговоркой" if out["correct"] else
                             "нет оговорки о том, что есть только общее правило" if not caveat else f"нет фактов: {missing}")
    else:
        if refused:
            out.update(false_refusal=True, reason="ложный отказ")
        else:
            ok, missing = facts_found(answer, q)
            out["correct"] = ok
            out["reason"] = "факты найдены" if ok else f"нет фактов: {missing}"
    return out


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def run_once(service, questions, pause):
    rows = []
    for q in questions:
        started = time.perf_counter()
        try:
            res = service.rag.ask(q["question"])
            err = None
        except LLMError as exc:
            res, err = None, f"{type(exc).__name__}"
        ms = round((time.perf_counter() - started) * 1000, 1)
        if res is None:
            rows.append({"id": q["id"], "type": q["type"], "error": err, "latency_ms": ms,
                         "judge": {"correct": False, "hallucination": False, "false_refusal": False,
                                   "cautious_refusal": False, "reason": f"ошибка LLM: {err}"}})
        else:
            rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "answer": res["answer"],
                         "refused": res["refused"], "partial": res["partial"], "status": res["status"],
                         "sources": res["sources"], "retrieval_hit": hit_at_k(q, res["retrieved_chunks"]),
                         "top_scores": [c["score"] for c in res["retrieved_chunks"]],
                         "latency_ms": ms, "judge": judge(q, res)})
        time.sleep(pause)
    return rows


def metrics(rows, questions):
    n = len(rows)
    retr = [r for r, q in zip(rows, questions) if q.get("expected_source")]
    refuse_q = [r for r in rows if r["type"] == "no_answer"]
    lat = [r["latency_ms"] for r in rows]
    correct = sum(r["judge"]["correct"] for r in rows)
    return {
        "questions": n,
        "correct": correct,
        "accuracy": round(correct / n, 3) if n else 0,
        "hit_at_k": round(sum(r.get("retrieval_hit", False) for r in retr) / len(retr), 3) if retr else None,
        "correct_refusals_on_no_answer": f"{sum(r['judge']['correct'] for r in refuse_q)}/{len(refuse_q)}",
        "hallucinations": sum(r["judge"]["hallucination"] for r in rows),
        "false_refusals": sum(r["judge"]["false_refusal"] for r in rows),
        "cautious_refusals": sum(r["judge"]["cautious_refusal"] for r in rows),
        "llm_errors": sum(1 for r in rows if r.get("error")),
        "avg_latency_ms": round(sum(lat) / n, 1) if n else 0,
        "p95_latency_ms": round(percentile(lat, 95), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="improved")
    ap.add_argument("--runs", type=int, default=1, help="число повторов (оценка разброса)")
    ap.add_argument("--pause", type=float, default=1.0, help="пауза между вопросами, с (rate limit)")
    args = ap.parse_args()

    from src.service import PharmaQCService
    service = PharmaQCService(args.preset)
    paths = service.config.section("paths")
    questions = json.loads(Path(paths["test_questions"]).read_text(encoding="utf-8"))
    os.makedirs(paths["eval_dir"], exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_metrics = []
    for run in range(1, args.runs + 1):
        print(f"\n=== {args.preset}: прогон {run}/{args.runs} ===")
        rows = run_once(service, questions, args.pause)
        m = metrics(rows, questions)
        all_metrics.append(m)
        for r in rows:
            mark = "OK " if r["judge"]["correct"] else "ERR"
            print(f"  {mark} Q{r['id']:<2} {r['type']:<9} {r['latency_ms']:>7.0f} мс  {r['judge']['reason']}")
        print("  ", m)
        out = Path(paths["eval_dir"]) / f"metrics_{args.preset}_{stamp}_run{run}.json"
        out.write_text(json.dumps({
            "preset": args.preset, "run": run, "timestamp": stamp,
            "config": {"retrieval": service.config.section("retrieval"), "llm": service.config.section("llm"),
                       "chunking": service.config.section("chunking")},
            "metrics": m, "detailed_results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    acc = [m["accuracy"] for m in all_metrics]
    summary = {
        "preset": args.preset, "runs": args.runs,
        "accuracy_mean": round(sum(acc) / len(acc), 3), "accuracy_min": min(acc), "accuracy_max": max(acc),
        "target_accuracy": TARGET_ACCURACY, "target_p95_ms": TARGET_P95_MS,
        "targets_met": min(acc) >= TARGET_ACCURACY and all(m["p95_latency_ms"] < TARGET_P95_MS for m in all_metrics),
        "per_run": all_metrics,
    }
    spath = Path(paths["eval_dir"]) / f"summary_{args.preset}_{stamp}.json"
    spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nИтог: accuracy mean={summary['accuracy_mean']} (min {summary['accuracy_min']}, max {summary['accuracy_max']}), "
          f"цели {'достигнуты' if summary['targets_met'] else 'НЕ достигнуты'}. Сводка: {spath}")


if __name__ == "__main__":
    main()
