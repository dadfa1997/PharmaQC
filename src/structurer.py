import hashlib
from typing import Dict, Any


class DocumentStructurer:
    def structure(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc_hash = hashlib.sha256(doc["raw_text"].encode('utf-8')).hexdigest()[:16]

        base_metadata = {
            "source": doc.get("source", ""),
            "file_type": doc.get("file_type", ""),
            "document_id": f"doc_{doc_hash}",
            "section": doc.get("category", "general"),
            "original_filename": doc.get("filename", ""),
        }

        skip_keys = {"raw_text", "source", "file_type", "filename"}
        for key, value in doc.items():
            if key not in skip_keys and key not in base_metadata:
                base_metadata[key] = value

        return {
            "text": doc["raw_text"],
            "metadata": base_metadata,
        }