"""FastAPI server exposing the TransactionRAGPipeline as a REST API."""

import base64
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import TransactionRAGPipeline
from config import DATA_DIR, OUTPUT_DIR

# ── Load data & init pipeline ──
df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
pipeline = TransactionRAGPipeline(df=df)

# ── FastAPI app ──
app = FastAPI(title="Tabular RAG Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve chart images
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


class QueryRequest(BaseModel):
    user_id: str
    prompt: str


class UserInfo(BaseModel):
    user_id: str
    user_name: str
    transaction_count: int


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/api/users")
async def get_users() -> list[UserInfo]:
    users = df.groupby("user_id").agg(
        user_name=("user_name", "first"),
        transaction_count=("user_id", "count"),
    ).reset_index()
    return [
        UserInfo(
            user_id=row["user_id"],
            user_name=row["user_name"],
            transaction_count=row["transaction_count"],
        )
        for _, row in users.iterrows()
    ]


@app.post("/api/query")
async def query(req: QueryRequest):
    result = pipeline.run(user_id=req.user_id, prompt=req.prompt)

    # Convert chart paths to URLs and encode images as base64
    viz_data = []
    for path in result.get("visualizations", []):
        if os.path.exists(path):
            filename = os.path.basename(path)
            # Also provide base64 for inline display
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            viz_data.append({
                "url": f"/output/{filename}",
                "base64": f"data:image/png;base64,{b64}",
                "filename": filename,
            })

    return {
        "user_name": result["user_name"],
        "response": result["response"],
        "data_summary": result["data_summary"],
        "visualizations": viz_data,
        "cache_hit": result["cache_hit"],
        "latency_ms": result["latency_ms"],
        "guardrail_flags": result["guardrail_flags"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
