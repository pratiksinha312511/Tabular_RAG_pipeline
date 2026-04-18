"""Central configuration for the Tabular RAG Pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model fallback chain (free models with tool calling support)
MODEL_CHAIN = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
]

# ── Guardrail Settings ──
MAX_PROMPT_LENGTH = 500
MAX_TOKENS_INPUT = 4000
MAX_TOKENS_OUTPUT = 2000
MAX_QUERY_HISTORY = 5
LLM_TIMEOUT_SECONDS = 30
CIRCUIT_BREAKER_THRESHOLD = 3

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
