from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os, torch
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    model.encode("Warming up...", show_progress_bar=False)

    yield  


app = FastAPI(lifespan=lifespan)

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        embedding = await run_in_threadpool(lambda: model.encode(request.text, show_progress_bar=False).tolist())
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
