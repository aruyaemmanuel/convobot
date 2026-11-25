from fastapi import FastAPI

app = FastAPI(title="convo-eval")


@app.get("/health")
def health():
    return {"status": "ok"}