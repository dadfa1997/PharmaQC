import json
import os
from typing import List, Dict, Any

class DataExporter:
    def export_jsonl(self, documents: List[Dict[str, Any]], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                
        print(f"[Exporter] Saved {len(documents)} documents to {filepath}")