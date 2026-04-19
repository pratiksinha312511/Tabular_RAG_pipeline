"""FastAPI server exposing the TransactionRAGPipeline as a REST API."""

import asyncio
import base64
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import TransactionRAGPipeline
from config import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger("server")

# ── Custom JSON encoder for numpy types ──

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def safe_json_response(content: dict, status_code: int = 200) -> JSONResponse:
    """JSONResponse that safely handles numpy types."""
    body = json.dumps(content, cls=NumpyEncoder, ensure_ascii=False)
    return JSONResponse(content=json.loads(body), status_code=status_code)

# ── Load data & init pipeline ──
# Use CSV_FILE from .env if set, otherwise auto-detect first .csv in data/
_csv_name = os.getenv("CSV_FILE", "").strip()
if _csv_name:
    _csv_path = Path(DATA_DIR) / _csv_name
    if not _csv_path.exists():
        raise FileNotFoundError(f"CSV_FILE={_csv_name} not found in {DATA_DIR}")
else:
    _csv_files = sorted(Path(DATA_DIR).glob("*.csv"))
    if not _csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}. "
            "Place a .csv file in data/ or set CSV_FILE in .env."
        )
    _csv_path = _csv_files[0]

df = pd.read_csv(_csv_path)
logger.info("Loaded %s (%d rows, %d cols)", _csv_path.name, len(df), len(df.columns))
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
pipeline = TransactionRAGPipeline(df=df)

# ── FastAPI app ──
app = FastAPI(title="Tabular RAG Pipeline", version="1.0.0")

# Thread pool for blocking pipeline calls
_executor = ThreadPoolExecutor(max_workers=2)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Filter noisy VS Code WebSocket probe logs ──
class _WSProbeFilter(logging.Filter):
    """Suppress 'connection open/closed' and WebSocket routing-trace log lines."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "routing-trace" in msg:
            return False
        if msg.strip() in ("connection open", "connection closed",
                           "connection rejected (403 Forbidden)"):
            return False
        return True

logging.getLogger("uvicorn.access").addFilter(_WSProbeFilter())
logging.getLogger("uvicorn.error").addFilter(_WSProbeFilter())

# ── WebSocket catch-all (silences VS Code Simple Browser / Live Preview probes) ──
@app.websocket("/ws/{path:path}")
async def ws_catchall(websocket: WebSocket, path: str):
    """Accept and immediately close any WebSocket the IDE/browser tries to open."""
    try:
        await websocket.accept()
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


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
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, lambda: pipeline.run(user_id=req.user_id, prompt=req.prompt)
        )
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        traceback.print_exc()
        return safe_json_response({
            "user_name": None,
            "response": "An internal error occurred. Please try again.",
            "data_summary": {},
            "visualizations": [],
            "cache_hit": False,
            "latency_ms": 0,
            "guardrail_flags": ["server_error"],
        }, status_code=200)  # Return 200 so frontend can read the JSON

    # Convert chart paths to URLs and encode images as base64
    viz_data = []
    for item in result.get("visualizations", []):
        if isinstance(item, dict):
            path = item.get("path", "")
            explanation = item.get("explanation", "")
        else:
            path = item
            explanation = ""
        try:
            if path and os.path.exists(path):
                filename = os.path.basename(path)
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                viz_data.append({
                    "url": f"/output/{filename}",
                    "base64": f"data:image/png;base64,{b64}",
                    "filename": filename,
                    "explanation": explanation,
                })
        except Exception as exc:
            logger.warning(f"Failed to encode chart {path}: {exc}")

    return safe_json_response({
        "user_name": result.get("user_name"),
        "response": result.get("response", ""),
        "data_summary": result.get("data_summary", {}),
        "visualizations": viz_data,
        "cache_hit": result.get("cache_hit", False),
        "latency_ms": result.get("latency_ms", 0),
        "guardrail_flags": result.get("guardrail_flags", []),
    })


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """SSE endpoint that streams the LLM response token-by-token."""

    def _encode_charts(items: list) -> list:
        """Convert chart items (path strings or dicts with path+explanation) to base64."""
        charts = []
        for item in items:
            # Support both plain path strings and {"path": ..., "explanation": ...} dicts
            if isinstance(item, dict):
                path = item.get("path", "")
                explanation = item.get("explanation", "")
            else:
                path = item
                explanation = ""
            try:
                if path and os.path.exists(path):
                    filename = os.path.basename(path)
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    charts.append({
                        "url": f"/output/{filename}",
                        "base64": f"data:image/png;base64,{b64}",
                        "filename": filename,
                        "explanation": explanation,
                    })
            except Exception as exc:
                logger.warning(f"Failed to encode chart {path}: {exc}")
        return charts

    def _run_pipeline_stream(q: Queue):
        """Run the streaming pipeline in a thread, putting events into a queue."""
        try:
            for event in pipeline.run_stream(user_id=req.user_id, prompt=req.prompt):
                # Handle charts: convert paths to base64 inline
                if event.get("event") == "charts":
                    paths = event["data"].get("visualizations", [])
                    event["data"]["visualizations"] = _encode_charts(paths)
                # Handle done: sanitize numpy in data_summary
                if event.get("event") == "done" and "data_summary" in event.get("data", {}):
                    raw = json.dumps(event["data"]["data_summary"], cls=NumpyEncoder)
                    event["data"]["data_summary"] = json.loads(raw)
                q.put(event)
        except Exception as exc:
            logger.error(f"Stream pipeline error: {exc}")
            traceback.print_exc()
            q.put({"event": "error", "data": {"message": str(exc)}})
        finally:
            q.put(None)  # sentinel

    async def _sse_generator():
        """Async generator that reads from the queue and yields SSE lines."""
        q: Queue = Queue()
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_executor, _run_pipeline_stream, q)

        while True:
            # Poll the queue without blocking the event loop
            try:
                event = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: q.get(timeout=0.5)),
                    timeout=120,
                )
            except (Empty, asyncio.TimeoutError):
                continue

            if event is None:
                break

            event_type = event.get("event", "message")
            data_str = json.dumps(event.get("data", {}), cls=NumpyEncoder, ensure_ascii=False)
            yield f"event: {event_type}\ndata: {data_str}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/conversation/clear")
async def clear_conversation(req: QueryRequest):
    """Clear conversation history for a user (new chat)."""
    pipeline.cache.clear_conversation(req.user_id)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
