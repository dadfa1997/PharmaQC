from pathlib import Path

from src.config_loader import ConfigLoader

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def test_single_main_config_file():
    assert sorted(p.name for p in CONFIG_DIR.glob("*.yaml")) == ["app.yaml"]


def test_baseline_preset_overrides_retrieval_only():
    base, improved = ConfigLoader(), ConfigLoader("baseline")
    assert improved.get("retrieval", "top_k") == 3
    assert improved.get("retrieval", "similarity_threshold") == 0.0
    assert improved.get("llm", "model") == base.get("llm", "model")
    assert improved.get("chunking") == base.get("chunking")


def test_system_prompt_loaded_from_file():
    prompt = ConfigLoader().get("prompt", "system_prompt")
    assert "<context>" in prompt and "no_info" in prompt


def test_indexer_config_shape():
    cfg = ConfigLoader().indexer_config()
    assert cfg["vector_store"]["dimensions"] == 768
    assert set(cfg["paths"]) == {"input_jsonl", "validation_file", "search_results_file", "manifest_file"}


def test_prefixes_configured():
    c = ConfigLoader()
    assert c.get("embedding", "query_prefix") == "query: "
    assert c.get("embedding", "passage_prefix") == "passage: "
