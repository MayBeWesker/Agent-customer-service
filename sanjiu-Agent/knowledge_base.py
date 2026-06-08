from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import config
from storage import load_json
from vector_store import VectorStoreService


@dataclass
class MedicationMatch:
    name: str
    spec: str
    usage: str
    frequency: str
    times: list[str]
    source_text: str
    confidence: float
    raw_doc: dict[str, Any]


def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    md5_obj = hashlib.md5()
    md5_obj.update(input_str.encode(encoding=encoding))
    return md5_obj.hexdigest()


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = load_json(config.KNOWLEDGE_PATH, default=[])
        self.vector_store = VectorStoreService(self.documents)

    def list_documents(self) -> list[dict[str, Any]]:
        return self.documents

    def find_explicit_mentions(self, text: str) -> list[dict[str, Any]]:
        query = text.strip().lower()
        if not query:
            return []
        matched: list[dict[str, Any]] = []
        for document in self.documents:
            candidates = [document.get("name", "")] + list(document.get("aliases", []))
            if any(candidate and candidate.lower() in query for candidate in candidates):
                matched.append(document)
        return matched

    def normalize_medication_name(self, text: str) -> str:
        explicit = self.find_explicit_mentions(text)
        if explicit:
            return explicit[0].get("name", text)
        stripped = text.strip()
        if stripped == "999":
            return config.DEFAULT_MEDICATION_NAME
        return stripped

    def get_document_by_name(self, name: str) -> dict[str, Any] | None:
        target = name.strip().lower()
        for document in self.documents:
            if document.get("name", "").lower() == target:
                return document
            for alias in document.get("aliases", []):
                if str(alias).lower() == target:
                    return document
        return None

    def upload_by_str(self, data: str, file_name: str) -> str:
        md5_hex = get_string_md5(data)
        for document in self.documents:
            if document.get("md5") == md5_hex:
                return "内容已存在于药品知识库中"

        payload = {
            "name": file_name.replace(".txt", ""),
            "aliases": [],
            "spec": "",
            "usage": data[:120],
            "summary": data,
            "side_effects": "待补充",
            "contraindications": "待补充",
            "missed_dose": "如漏服，请参考说明书或咨询医生/药师。",
            "interactions": "待补充",
            "md5": md5_hex,
            "source": file_name,
        }
        self.documents.append(payload)
        from storage import save_json

        save_json(config.KNOWLEDGE_PATH, self.documents)
        self.vector_store = VectorStoreService(self.documents)
        return "upload successfully"

    def match_medication_from_text(self, raw_text: str) -> MedicationMatch | None:
        query = raw_text.strip()
        if not query:
            return None
        explicit = self.find_explicit_mentions(query)
        if explicit:
            doc = explicit[0]
            frequency = doc.get("default_frequency", "每日1次")
            return MedicationMatch(
                name=doc.get("name", "未识别药品"),
                spec=doc.get("spec", ""),
                usage=doc.get("usage", ""),
                frequency=frequency,
                times=doc.get("default_times", config.DEFAULT_FREQUENCY_MAP.get(frequency, ["08:00"])),
                source_text=raw_text,
                confidence=1.0,
                raw_doc=doc,
            )
        result = self.vector_store.query(query, top_k=1)
        if not result:
            return None
        top = result[0]
        doc = top.payload
        frequency = doc.get("default_frequency", "每日1次")
        return MedicationMatch(
            name=doc.get("name", "未识别药品"),
            spec=doc.get("spec", ""),
            usage=doc.get("usage", ""),
            frequency=frequency,
            times=doc.get("default_times", config.DEFAULT_FREQUENCY_MAP.get(frequency, ["08:00"])),
            source_text=raw_text,
            confidence=round(top.score, 2),
            raw_doc=doc,
        )
