import re
import uuid
from rank_bm25 import BM25Okapi


DOCS: dict[str, dict] = {}
INDEX: dict[str, BM25Okapi] = {}

def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def retrieve(doc_id: str, query: str, k: int = 3) -> list[dict]:
    scores = INDEX[doc_id].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"idx": i, "score": float(scores[i]), "text": DOCS[doc_id]["chunks"][i]}
        for i in ranked
    ]

def chunk(text: str, size: int = 200) -> list[str]:
    # split on blank lines into paragraphs
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def ingest(name: str, text: str) -> tuple[str, int]:
    cs = chunk(text)
    doc_id = uuid.uuid4().hex[:8]
    DOCS[doc_id] = {"name": name, "chunks": cs}
    INDEX[doc_id] = BM25Okapi([tokenize(c) for c in cs])
    return doc_id, len(cs)