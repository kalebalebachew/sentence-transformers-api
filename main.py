from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer('all-mpnet-base-v2')

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    try:
        embedding = model.encode(request.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
