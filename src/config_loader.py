"""Единая загрузка конфигурации: config/app.yaml + пресет + системный промпт."""
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "app.yaml"


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


class ConfigLoader:
    """Читает app.yaml (путь можно задать PHARMAQC_CONFIG), применяет пресет, подгружает промпт."""

    def __init__(self, preset_name: Optional[str] = None, config_path: Optional[str] = None):
        path = Path(config_path or os.getenv("PHARMAQC_CONFIG") or DEFAULT_CONFIG)
        if not path.exists():
            raise FileNotFoundError(f"Конфигурация не найдена: {path}")
        self.config_path = path
        with open(path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f) or {}

        prompt_file = (self.config.get("prompt") or {}).get("system_prompt_file")
        if prompt_file:
            prompt_path = self._resolve(prompt_file)
            if not prompt_path.exists():
                raise FileNotFoundError(f"Промпт не найден: {prompt_path}")
            self.config["prompt"]["system_prompt"] = prompt_path.read_text(encoding="utf-8").strip()

        self.preset_name = None
        if preset_name:
            self.apply_preset(preset_name)

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute() or p.exists():
            return p
        return PROJECT_ROOT / p

    def apply_preset(self, preset_name: str) -> None:
        preset_path = PROJECT_ROOT / "config" / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Пресет не найден: {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = yaml.safe_load(f) or {}
        _merge(self.config, preset)
        self.preset_name = preset_name

    def get(self, *keys: str, default: Any = None) -> Any:
        """config.get("llm", "model") -> config["llm"]["model"]."""
        node: Any = self.config
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def section(self, name: str) -> Dict[str, Any]:
        return copy.deepcopy(self.config.get(name, {}))

    def indexer_config(self) -> Dict[str, Any]:
        """Конфиг в формате, который ожидает VectorStoreIndexer."""
        p = self.config["paths"]
        return {
            "vector_store": self.section("vector_store"),
            "paths": {
                "input_jsonl": p["embeddings_jsonl"],
                "validation_file": p["validation_file"],
                "search_results_file": p["search_results_file"],
                "manifest_file": p["manifest_file"],
            },
            "indexing": self.section("indexing"),
            "search": self.section("search"),
        }
