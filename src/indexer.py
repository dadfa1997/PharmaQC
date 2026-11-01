import json
import math
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings


class VectorStoreIndexer:
    def __init__(self, config: Dict[str, Any], config_path: str = ""):
        self.vs_cfg = config["vector_store"]
        self.paths = config["paths"]
        self.idx_cfg = config.get("indexing", {})
        self.search_cfg = config.get("search", {})

        self.collection_name = self.vs_cfg["collection_name"]
        self.persist_dir = self.vs_cfg["persist_directory"]
        self.distance = self.vs_cfg.get("distance_metric", "cosine")
        self.expected_dim = self.vs_cfg.get("dimensions")
        if self.expected_dim is None:
            raise ValueError("'dimensions' is required in config/app.yaml under 'vector_store'")
        self.recreate = self.vs_cfg.get("recreate_collection", True)
        self.batch_size = self.idx_cfg.get("batch_size", 100)

        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.paths["validation_file"]), exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None
        self.config_path = config_path
        self.config_snapshot = config

    # ── Validation ──────────────────────────────────────────────

    def load_and_validate(self):
        path = self.paths["input_jsonl"]
        if not os.path.exists(path):
            raise FileNotFoundError("Embeddings file not found: " + path)

        records = []
        errors = []

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append("Line " + str(i) + ": invalid JSON - " + str(e))
                    continue

                if not isinstance(rec, dict):
                    errors.append("Line " + str(i) + ": expected dict, got " + type(rec).__name__)
                    continue

                missing_top = False
                for field in ("text", "embedding", "metadata"):
                    if field not in rec:
                        errors.append("Line " + str(i) + ": missing field '" + field + "'")
                        missing_top = True
                        break
                if missing_top:
                    continue

                meta = rec["metadata"]
                if not isinstance(meta, dict):
                    errors.append("Line " + str(i) + ": metadata must be dict, got " + type(meta).__name__)
                    continue

                required_fields = (
                    "chunk_id",
                    "document_id",
                    "chunk_index",
                    "embedding_model",
                    "embedding_dimensions",
                )
                meta_ok = True
                for mf in required_fields:
                    if mf not in meta:
                        errors.append("Line " + str(i) + ": metadata missing '" + mf + "'")
                        meta_ok = False
                        break
                    val = meta[mf]
                    if val is None or (isinstance(val, str) and not val.strip()):
                        errors.append("Line " + str(i) + ": metadata field '" + mf + "' is empty")
                        meta_ok = False
                        break
                if not meta_ok:
                    continue

                emb = rec["embedding"]
                if not isinstance(emb, list):
                    errors.append("Line " + str(i) + ": embedding must be list, got " + type(emb).__name__)
                    continue

                if len(emb) == 0:
                    errors.append("Line " + str(i) + ": embedding is empty")
                    continue

                bad_elements = False
                for j, v in enumerate(emb):
                    if not isinstance(v, (int, float)):
                        errors.append("Line " + str(i) + ": embedding[" + str(j) + "] must be number, got " + type(v).__name__)
                        bad_elements = True
                        break
                if bad_elements:
                    continue

                declared_dim = meta["embedding_dimensions"]
                actual_dim = len(emb)
                if declared_dim != actual_dim:
                    errors.append("Line " + str(i) + ": embedding_dimensions=" + str(declared_dim) + " != actual=" + str(actual_dim))
                    continue

                if actual_dim != self.expected_dim:
                    errors.append("Line " + str(i) + ": vector dim=" + str(actual_dim) + " != configured=" + str(self.expected_dim))
                    continue

                if any(math.isnan(v) or math.isinf(v) for v in emb):
                    errors.append("Line " + str(i) + ": NaN/Infinity in embedding")
                    continue

                records.append(rec)

        total_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                if raw_line.strip():
                    total_lines += 1

        if errors:
            print("[Indexer] Validation errors:")
            for err in errors:
                print("  " + err)
            print("[Indexer] " + str(len(errors)) + " error(s) found in " + str(total_lines) + " records")

        print("[Indexer] " + str(len(records)) + "/" + str(total_lines) + " valid records loaded (configured dimension=" + str(self.expected_dim) + ")")
        return records, total_lines

    # ── Index creation ──────────────────────────────────────────

    def create_or_get_collection(self):
        if self.recreate:
            print("[Indexer] Recreating collection '" + self.collection_name + "' (dim=" + str(self.expected_dim) + ", metric=" + self.distance + ")")
            existing = self.client.list_collections()
            if self.collection_name in [c.name for c in existing]:
                self.client.delete_collection(name=self.collection_name)
                print("[Indexer] Deleted existing collection '" + self.collection_name + "'")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
            )
            print("[Indexer] Created new collection '" + self.collection_name + "'")
        else:
            existing = self.client.list_collections()
            existing_names = [c.name for c in existing]

            if self.collection_name not in existing_names:
                print("[Indexer] Collection '" + self.collection_name + "' does not exist, creating it")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance},
                )
                return

            self.collection = self.client.get_collection(name=self.collection_name)
            print("[Indexer] Using existing collection '" + self.collection_name + "'")

            col_meta = self.collection.metadata or {}
            actual_metric = col_meta.get("hnsw:space", "unknown")

            if actual_metric != self.distance:
                raise ValueError(
                    "Collection '" + self.collection_name + "' has metric '" + str(actual_metric) + "', "
                    "but config requires '" + self.distance + "'. "
                    "Set recreate_collection: true to recreate with correct parameters."
                )

            count = self.collection.count()
            if count > 0:
                sample = self.collection.get(limit=1, include=["embeddings"])
                emb_list = sample.get("embeddings") if sample is not None else None
                if emb_list is not None and len(emb_list) > 0:
                    actual_dim = len(emb_list[0])
                    if actual_dim != self.expected_dim:
                        raise ValueError(
                            "Collection '" + self.collection_name + "' has vectors of dimension " + str(actual_dim) + ", "
                            "but config requires " + str(self.expected_dim) + ". "
                            "Set recreate_collection: true to recreate with correct parameters."
                        )
                    print("[Indexer] Parameters verified: dim=" + str(actual_dim) + ", metric=" + str(actual_metric))
                else:
                    print("[Indexer] Metric verified: " + str(actual_metric) + " (no vectors to check dimension)")
            else:
                print("[Indexer] Metric verified: " + str(actual_metric) + " (empty collection)")

    # ── Upload ──────────────────────────────────────────────────

    def upload(self, records: List[Dict[str, Any]]):
        total = len(records)
        uploaded = 0

        for start in range(0, total, self.batch_size):
            batch = records[start : start + self.batch_size]
            ids = [r["metadata"]["chunk_id"] for r in batch]
            embeddings = [r["embedding"] for r in batch]
            documents = [r["text"] for r in batch]
            metadatas = [r["metadata"] for r in batch]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            uploaded += len(batch)
            print("[Indexer] Uploaded " + str(uploaded) + "/" + str(total))

        return uploaded

    # ── Validation check ───────────────────────────────────────

    def validate_index(self, total_input_count: int, valid_count: int) -> Dict[str, Any]:
        count = self.collection.count()

        sample_size = min(5, count) if count > 0 else 0
        sample = None
        if sample_size > 0:
            sample = self.collection.get(
                limit=sample_size,
                include=["documents", "metadatas", "embeddings"],
            )

        checks = {}

        # 1. Совпадение количества
        checks["counts_match"] = valid_count == count and total_input_count == valid_count

        # 2. Наличие payload (document + metadata вместе)
        docs_list = sample.get("documents") if sample is not None else None
        meta_list = sample.get("metadatas") if sample is not None else None
        has_docs = docs_list is not None and len(docs_list) > 0
        has_meta_raw = meta_list is not None and len(meta_list) > 0
        checks["has_payload"] = has_docs and has_meta_raw

        # 3. Текст непустой
        checks["text_not_empty"] = has_docs and all(isinstance(d, str) and len(d.strip()) > 0 for d in docs_list)

        # 4. Наличие embedding и проверка размерности
        emb_list = sample.get("embeddings") if sample is not None else None
        has_emb = emb_list is not None and len(emb_list) > 0
        checks["has_embeddings"] = has_emb
        checks["dimensions_correct"] = False
        if has_emb:
            dims = set(len(e) for e in emb_list)
            checks["dimensions_correct"] = (len(dims) == 1) and (self.expected_dim in dims)
            checks["actual_dimensions"] = list(dims)

        # 5. Обязательные поля metadata
        required_meta_fields = ("chunk_id", "document_id", "chunk_index", "embedding_model", "embedding_dimensions")
        checks["has_required_metadata"] = False
        if has_meta_raw:
            all_have_fields = True
            for meta in meta_list:
                if not isinstance(meta, dict):
                    all_have_fields = False
                    break
                for mf in required_meta_fields:
                    if mf not in meta or meta[mf] is None or (isinstance(meta[mf], str) and not meta[mf].strip()):
                        all_have_fields = False
                        break
                if not all_have_fields:
                    break
            checks["has_required_metadata"] = all_have_fields

        # 6. Пробный поисковый запрос
        checks["search_works"] = False
        checks["search_returns_results"] = False
        if has_emb:
            try:
                test_query = self.collection.query(
                    query_embeddings=[emb_list[0]],
                    n_results=1,
                    include=["documents", "metadatas", "distances"],
                )
                checks["search_works"] = True
                tq_ids = test_query.get("ids") if test_query is not None else None
                has_ids = tq_ids is not None and len(tq_ids) > 0 and len(tq_ids[0]) > 0
                checks["search_returns_results"] = has_ids
            except Exception as e:
                checks["search_works"] = False
                checks["search_error"] = str(e)

        # Итоговый статус
        check_keys = (
            "counts_match",
            "has_payload",
            "text_not_empty",
            "has_embeddings",
            "dimensions_correct",
            "has_required_metadata",
            "search_works",
            "search_returns_results",
        )
        all_passed = all(checks.get(k) is True for k in check_keys)

        result = {
            "total_input_embeddings": total_input_count,
            "valid_embeddings_loaded": valid_count,
            "skipped_embeddings": total_input_count - valid_count,
            "index_points_count": count,
            "sample_size_checked": sample_size,
            "checks": checks,
            "all_checks_passed": all_passed,
        }

        with open(self.paths["validation_file"], "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        status = "PASSED" if all_passed else "FAILED"
        passed_count = sum(1 for k in check_keys if checks.get(k) is True)
        print("[Indexer] Validation: " + status + " (" + str(passed_count) + "/" + str(len(check_keys)) + " checks passed)")
        for check_name in check_keys:
            check_val = checks.get(check_name)
            mark = "+" if check_val is True else "-"
            print("  [" + mark + "] " + check_name + ": " + str(check_val))
        if "actual_dimensions" in checks:
            print("  [i] actual_dimensions: " + str(checks["actual_dimensions"]))
        if "search_error" in checks:
            print("  [!] search_error: " + str(checks["search_error"]))

        return result

    # ── Test search ─────────────────────────────────────────────

    def test_search(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = min(self.search_cfg.get("test_queries_count", 3), len(records))
        top_k = self.search_cfg.get("top_k", 5)
        results = []

        for i in range(n):
            rec = records[i]
            query_emb = rec["embedding"]
            query_text = rec["text"]
            query_doc_id = rec["metadata"].get("document_id", "")
            query_chunk_id = rec["metadata"]["chunk_id"]

            search_res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            sr_ids = search_res.get("ids", [[]])[0]
            sr_distances = search_res.get("distances", [[]])[0]
            sr_documents = search_res.get("documents", [[]])[0]
            sr_metadatas = search_res.get("metadatas", [[]])[0]

            for j in range(len(sr_ids)):
                raw_dist = sr_distances[j]
                if self.distance == "cosine":
                    # cosine distance: 0 = идентичны, 2 = противоположны → score = 1 - dist
                    score = round(1.0 - raw_dist, 4)
                elif self.distance == "l2":
                    # l2 distance: 0 = идентичны, больше = хуже → score = 1 / (1 + dist)
                    score = round(1.0 / (1.0 + raw_dist), 4)
                elif self.distance == "ip":
                    # inner product: больше = лучше, но Chroma возвращает negative IP как distance
                    # score = -distance (для negative IP) или raw (если уже положительный)
                    score = round(-raw_dist if raw_dist < 0 else raw_dist, 4)
                else:
                    score = round(raw_dist, 4)
                text_preview = sr_documents[j] if sr_documents[j] else ""
                hit_meta = sr_metadatas[j] if sr_metadatas[j] else {}
                hits.append({
                    "id": sr_ids[j],
                    "score": score,
                    "text": text_preview,
                    "metadata": hit_meta,
                })

            first_is_self = hits[0]["id"] == query_chunk_id if hits else False

            # Оценка релевантности: проверяем top-k результаты помимо первого
            relevance_signals = []

            # Сигнал 1: первый результат — сам запросный чанк
            if first_is_self:
                relevance_signals.append("first_hit_is_self")

            # Сигнал 2: другие результаты из того же документа
            same_doc_hits = 0
            for hit in hits:
                hit_doc_id = hit.get("metadata", {}).get("document_id", "")
                if hit_doc_id == query_doc_id and hit["id"] != query_chunk_id:
                    same_doc_hits += 1
            if same_doc_hits > 0:
                relevance_signals.append("same_document_hits=" + str(same_doc_hits))

            # Сигнал 3: все результаты имеют ненулевой score (для cosine: > 0)
            all_positive_scores = all(h["score"] > 0 for h in hits) if hits else False
            if all_positive_scores:
                relevance_signals.append("all_positive_scores")

            # Сигнал 4: score падает монотонно (результаты упорядочены по релевантности)
            scores_descending = True
            for j in range(1, len(hits)):
                if hits[j]["score"] > hits[j - 1]["score"]:
                    scores_descending = False
                    break
            if scores_descending and len(hits) > 1:
                relevance_signals.append("scores_descending")

            # looks_relevant = есть хотя бы 2 сигнала релевантности
            looks_relevant = len(relevance_signals) >= 2

            entry = {
                "query_chunk_id": query_chunk_id,
                "query_text": query_text,
                "top_k": top_k,
                "results": hits,
                "first_result_is_self": first_is_self,
                "relevance_signals": relevance_signals,
                "looks_relevant": looks_relevant,
            }
            results.append(entry)
            print("[Indexer] Test query " + str(i + 1) + "/" + str(n) + ": self=" + str(first_is_self) + ", relevant=" + str(looks_relevant) + ", signals=" + str(relevance_signals))

        with open(self.paths["search_results_file"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    # ── Manifest ────────────────────────────────────────────────

    def save_manifest(self, input_count: int, uploaded_count: int, validation: Dict):
        manifest = {
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_path": self.config_path,
            "config_snapshot": self.config_snapshot,
            "vector_store_type": self.vs_cfg["type"],
            "collection_name": self.collection_name,
            "distance_metric": self.distance,
            "dimensions": self.expected_dim,
            "input_embeddings_count": input_count,
            "uploaded_points_count": uploaded_count,
            "validation_passed": validation.get("all_checks_passed", False),
            "validation_checks": validation.get("checks", {}),
            "validation_file": self.paths["validation_file"],
            "search_results_file": self.paths["search_results_file"],
        }

        with open(self.paths["manifest_file"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print("[Indexer] Manifest saved: run_id=" + manifest["run_id"])
        return manifest

    # ── Full pipeline ───────────────────────────────────────────

    def run(self):
        print("=" * 50)
        print("VECTOR STORE INDEXING PIPELINE")
        print("=" * 50)

        records, total_input_count = self.load_and_validate()
        if not records:
            print("[Indexer] No valid records. Aborting.")
            return

        has_validation_errors = len(records) < total_input_count

        self.create_or_get_collection()
        uploaded = self.upload(records)
        validation = self.validate_index(total_input_count, len(records))

        if has_validation_errors:
            validation["all_checks_passed"] = False
            validation["note"] = str(total_input_count - len(records)) + " input records were invalid and skipped"
            with open(self.paths["validation_file"], "w", encoding="utf-8") as f:
                json.dump(validation, f, indent=2, ensure_ascii=False)
            print("[Indexer] WARNING: " + str(total_input_count - len(records)) + " records skipped due to validation errors")

        self.test_search(records)
        self.save_manifest(len(records), uploaded, validation)

        print("=" * 50)
        print("INDEXING COMPLETE")
        print("=" * 50)