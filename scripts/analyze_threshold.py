"""Анализ оценок схожести: помогает выбрать similarity_threshold на реальных данных.

  python scripts/analyze_threshold.py

Для каждого вопроса печатает лучший балл «релевантных» чанков (содержат ключевой факт из
required_facts) и лучший балл остальных. Если диапазоны перекрываются, порог не разделяет
релевантное и нерелевантное, и отказ нужно решать проверкой опоры в контексте (так и сделано).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import ConfigLoader  # noqa: E402
from src.rag import normalize_text  # noqa: E402


def main():
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    cfg = ConfigLoader()
    model = SentenceTransformer(cfg.get("embedding", "model_name"), device=cfg.get("embedding", "device", default="cpu"))
    client = chromadb.PersistentClient(path=cfg.get("vector_store", "persist_directory"),
                                       settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(cfg.get("vector_store", "collection_name"))
    prefix = cfg.get("embedding", "query_prefix", default="query: ")
    questions = json.loads(Path(cfg.get("paths", "test_questions")).read_text(encoding="utf-8"))

    report = []
    print(f"{'Q':>2} {'тип':<9} {'top1':>6} {'best relevant':>14} {'best other':>11}")
    for q in questions:
        emb = model.encode([prefix + q["question"]], normalize_embeddings=True)[0].tolist()
        res = col.query(query_embeddings=[emb], n_results=10, include=["documents", "distances"])
        facts = [normalize_text(a) for g in (q.get("required_facts") or [])[:1] for a in g]
        rel, other = [], []
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            score = round(1 - dist, 4)
            (rel if facts and any(f in normalize_text(doc) for f in facts) else other).append(score)
        top1 = max(rel + other)
        print(f"{q['id']:>2} {q['type']:<9} {top1:>6.3f} {max(rel) if rel else float('nan'):>14.3f} {max(other) if other else float('nan'):>11.3f}")
        report.append({"id": q["id"], "type": q["type"], "top1": top1, "best_relevant": max(rel) if rel else None,
                       "best_other": max(other) if other else None})

    out = Path(cfg.get("paths", "eval_dir")) / "threshold_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
