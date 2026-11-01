import os
import json
from pathlib import Path
from typing import List, Dict, Any

class DataLoader:
    SUPPORTED_EXTENSIONS = {'.txt', '.json'}

    def load(self, directory: str, metadata_path: str = "data/documents_metadata.json") -> List[Dict[str, Any]]:
        documents = []
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory {directory} not found")

        docs_meta = self.load_documents_metadata(metadata_path)

        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                print(f"[Loader] Skipping unsupported file: {file_path.name}")
                continue

            try:
                content = self._read_file(file_path)
                file_meta = docs_meta.get(file_path.name, {})
                
                documents.append({
                    "raw_text": content,
                    "source": str(file_path),
                    "file_type": file_path.suffix.lstrip('.'),
                    "filename": file_path.name,
                    **file_meta  # document_name, category, version
                })
                print(f"[Loader] Loaded: {file_path.name}")
            except Exception as e:
                print(f"[Loader] Error loading {file_path.name}: {e}")

        return documents

    def _read_file(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.json':
                data = json.load(f)
                # Простая сериализация JSON в читаемый текст для RAG
                return self._json_to_text(data)
            else:
                return f.read()

    def _json_to_text(self, data: Any, prefix: str = "") -> str:
        """Рекурсивное преобразование JSON в плоский текст"""
        lines = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_prefix = f"{prefix}{k}: " if prefix else f"{k}: "
                lines.append(self._json_to_text(v, new_prefix))
        elif isinstance(data, list):
            for item in data:
                lines.append(self._json_to_text(item, prefix))
        else:
            lines.append(f"{prefix}{str(data)}")
        return "\n".join(lines)

    def load_documents_metadata(self, path: str = "data/documents_metadata.json") -> Dict[str, Dict]:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)