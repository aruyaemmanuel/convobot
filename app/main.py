import time
import uuid

from fastapi import FastAPI, HTTPException
from groq import APIConnectionError, APIStatusError
from pydantic import BaseModel

from . import llm, store, logger

app = FastAPI(title="convo-eval")

SESSIONS: dict[str, list[dict]] = {}

SYSTEM = (
    "You are a business assistant. Answer ONLY from the provided context and the "
    "conversation so far. If the answer is not in the context, say you don't have "
    "that information. Be concise."
)


class IngestIn(BaseModel):
    name: str
    text: str


class SearchIn(BaseModel):
    doc_id: str
    query: str


class ChatIn(BaseModel):
    doc_id: str
    message: str
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(store.DOCS)}


@app.post("/ingest")
def ingest(body: IngestIn):
    doc_id, n = store.ingest(body.name, body.text)
    return {"doc_id": doc_id, "chunks": n}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    if doc_id not in store.DOCS:
        raise HTTPException(404, "unknown doc_id")
    return store.DOCS[doc_id]


@app.post("/search")
def search(body: SearchIn):
    if body.doc_id not in store.DOCS:
        raise HTTPException(404, "unknown doc_id")
    return {"results": store.retrieve(body.doc_id, body.query)}


@app.post("/chat")
def chat(body: ChatIn):
    if body.doc_id not in store.DOCS:
        raise HTTPException(404, "unknown doc_id")

    sid = body.session_id or uuid.uuid4().hex[:8]
    history = SESSIONS.setdefault(sid, [])

    # include the previous user message so follow-ups still retrieve well
    prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    chunks = store.retrieve(body.doc_id, f"{prev_user} {body.message}".strip())
    context = "\n---\n".join(c["text"] for c in chunks)

    messages = [{"role": "system", "content": f"{SYSTEM}\n\nContext:\n{context}"}]
    messages += history + [{"role": "user", "content": body.message}]

    t0 = time.time()
    try:
        answer, usage = llm.complete(messages)
    except APIStatusError as e:
        raise HTTPException(502, f"LLM error {e.status_code}: {e.message}")
    except APIConnectionError:
        raise HTTPException(503, "could not reach the LLM provider")
    latency_ms = int((time.time() - t0) * 1000)

    history += [
        {"role": "user", "content": body.message},
        {"role": "assistant", "content": answer},
    ]
    logger.log_turn({
        "session_id": sid,
        "doc_id": body.doc_id,
        "turn": len(history) // 2,
        "question": body.message,
        "retrieved": [c["idx"] for c in chunks],
        "answer": answer,
        "latency_ms": latency_ms,
        "usage": usage,
        "model": llm.MODEL,
    })
    return {
        "session_id": sid,
        "answer": answer,
        "retrieved": [c["idx"] for c in chunks],
        "latency_ms": latency_ms,
        "usage": usage,
    }