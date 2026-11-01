"""Чанкер по структуре регламента.

Документ состоит из разделов («2. Приёмка сырья») и нумерованных пунктов («2.3. Маркировка …»).
Пункт — минимальная смысловая единица: он не разрезается, если помещается в chunk_size, и
соседние короткие пункты одного раздела объединяются. В каждый чанк добавляются название
документа и заголовок раздела. Текст не теряется: все строки документа попадают в чанки.
"""
import re
from typing import Any, Dict, List, Optional

_SECTION_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")            # «2. Приёмка сырья»
_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(\S.*)$")   # «2.3. Текст …» или «2.3 Текст …»
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


class TextChunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80, min_chunk_size: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── публичный API ────────────────────────────────────────────

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        document_id = metadata.get("document_id", "")
        document_name = metadata.get("document_name", "")

        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        chunks: List[Dict[str, Any]] = []
        for section in self._parse_sections(text):
            for body, clauses in self._section_chunks(section):
                header = f"{section['number']}. {section['title']}" if section["number"] else section["title"]
                prefix = f"[{document_name}] {header}".strip() if header else f"[{document_name}]"
                chunk_text = f"{prefix}\n{body}"
                idx = len(chunks)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": f"{document_id}_chunk{idx}",
                        "chunk_index": idx,
                        "chunk_size": len(chunk_text),
                        "chunk_overlap": self.chunk_overlap,
                        "section_number": section["number"],
                        "section_title": section["title"],
                        "clauses": clauses,
                    },
                })
        return chunks

    # ── разбор структуры ─────────────────────────────────────────

    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """Возвращает разделы: {number, title, blocks: [{clause, lines}]}.

        Строки до первого раздела (название документа, версия) в чанки не идут:
        название документа уже есть в префиксе каждого чанка, версия — в метаданных.
        """
        sections: List[Dict[str, Any]] = []
        section: Optional[Dict[str, Any]] = None
        block: Optional[Dict[str, Any]] = None

        for raw in text.split("\n"):
            line = raw.strip()
            if not line:
                block = None
                continue

            m_clause = _CLAUSE_RE.match(line)
            m_section = _SECTION_RE.match(line)

            if m_clause and section is not None:
                block = {"clause": m_clause.group(1), "lines": [line]}
                section["blocks"].append(block)
            elif m_section and not m_clause:
                section = {"number": m_section.group(1), "title": m_section.group(2).strip(), "blocks": []}
                sections.append(section)
                block = None
            elif section is None:
                continue  # шапка документа
            elif block is not None:
                block["lines"].append(line)
            else:
                block = {"clause": "", "lines": [line]}
                section["blocks"].append(block)

        return [s for s in sections if s["blocks"]]

    # ── формирование чанков раздела ──────────────────────────────

    def _section_chunks(self, section: Dict[str, Any]):
        """Генерирует (текст, «пункты») для раздела."""
        pending: List[Dict[str, Any]] = []
        pending_len = 0
        out: List[Any] = []

        def flush():
            nonlocal pending, pending_len
            if pending:
                out.append(("\n".join("\n".join(b["lines"]) for b in pending), self._clause_range(pending)))
            pending, pending_len = [], 0

        for block in section["blocks"]:
            block_text = "\n".join(block["lines"])
            if len(block_text) > self.chunk_size:
                flush()
                for part in self._split_long_block(block):
                    out.append((part, block["clause"]))
                continue
            extra = len(block_text) + (1 if pending else 0)
            if pending and pending_len + extra > self.chunk_size:
                flush()
                extra = len(block_text)
            pending.append(block)
            pending_len += extra
        flush()

        # Слишком короткий последний чанк раздела присоединяем к предыдущему (текст не теряем)
        if len(out) >= 2 and len(out[-1][0]) < self.min_chunk_size:
            prev_text, prev_clauses = out[-2]
            last_text, last_clauses = out[-1]
            out[-2:] = [(prev_text + "\n" + last_text, self._join_clauses(prev_clauses, last_clauses))]
        return out

    def _split_long_block(self, block: Dict[str, Any]) -> List[str]:
        """Режет длинный пункт по строкам списка и предложениям, повторяя заголовок пункта."""
        lead = block["lines"][0]
        units: List[str] = []
        for line in block["lines"][1:] if len(block["lines"]) > 1 else []:
            units.extend(self._split_line(line))
        if len(block["lines"]) == 1:  # длинный однострочный пункт
            units = self._split_line(lead)
            lead = ""

        lead_prefix = (lead[:150] + "\n") if lead else ""
        budget = max(self.chunk_size - len(lead_prefix), self.chunk_size // 2)

        parts: List[str] = []
        current: List[str] = []
        current_len = 0
        for unit in units:
            add = len(unit) + (1 if current else 0)
            if current and current_len + add > budget:
                parts.append(lead_prefix + "\n".join(current))
                tail, tail_len = [], 0
                for u in reversed(current):
                    if tail_len + len(u) + 1 > self.chunk_overlap:
                        break
                    tail.insert(0, u)
                    tail_len += len(u) + 1
                current, current_len = tail, tail_len
                add = len(unit) + (1 if current else 0)
            current.append(unit)
            current_len += add
        if current:
            parts.append(lead_prefix + "\n".join(current))
        return parts

    def _split_line(self, line: str) -> List[str]:
        if len(line) <= self.chunk_size:
            return [line]
        pieces = [s.strip() for s in _SENTENCE_RE.split(line) if s.strip()]
        result: List[str] = []
        for piece in pieces:
            while len(piece) > self.chunk_size:  # крайний случай: предложение длиннее чанка
                result.append(piece[: self.chunk_size])
                piece = piece[self.chunk_size - self.chunk_overlap:]
            result.append(piece)
        return result

    @staticmethod
    def _clause_range(blocks: List[Dict[str, Any]]) -> str:
        nums = [b["clause"] for b in blocks if b["clause"]]
        if not nums:
            return ""
        return nums[0] if len(nums) == 1 else f"{nums[0]}–{nums[-1]}"

    @staticmethod
    def _join_clauses(a: str, b: str) -> str:
        if not a:
            return b
        if not b:
            return a
        return f"{a.split('–')[0]}–{b.split('–')[-1]}"
