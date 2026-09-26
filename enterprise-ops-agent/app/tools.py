"""Deterministic tools used by the enterprise incident agent MVP."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def search_logs(query: str, service: str | None = None, limit: int = 20) -> list[dict]:
    """Return matching structured logs without asking an LLM to interpret raw data."""
    needle = query.strip().lower()
    records: list[dict] = []
    with (DATA_DIR / "logs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if service and record.get("service") != service:
                continue
            haystack = json.dumps(record, ensure_ascii=False).lower()
            if not needle or needle in haystack:
                records.append(record)
            if len(records) >= limit:
                break
    return records


def get_metric_snapshot(service: str | None = None) -> dict:
    """Return the latest metric snapshot for the demo service."""
    snapshot = json.loads((DATA_DIR / "metrics.json").read_text(encoding="utf-8"))
    if service and snapshot.get("service") != service:
        return {"service": service, "found": False}
    return {**snapshot, "found": True}


def search_docs(query: str, limit: int = 5) -> list[dict]:
    """Small lexical baseline; it is intentionally deterministic before vector RAG."""
    terms = {term.lower() for term in query.split() if term.strip()}
    hits: list[dict] = []
    for path in sorted((DATA_DIR / "documents").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        score = sum(text.lower().count(term) for term in terms)
        if score:
            hits.append({"source": path.name, "score": score, "content": text})
    return sorted(hits, key=lambda item: item["score"], reverse=True)[:limit]
