from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import store

app = FastAPI(title="convo-eval")


class SearchIn(BaseModel):
    doc_id: str
    query: str

class IngestIn(BaseModel):
    name: str
    text: str


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