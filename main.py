#!/usr/bin/env python

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class InputText(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embeddings: List[float]


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(input_text: InputText) -> EmbeddingResponse:
    embeddings = model.encode(input_text.text)
    return EmbeddingResponse(embeddings=embeddings.tolist())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
