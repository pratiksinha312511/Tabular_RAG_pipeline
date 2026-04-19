"""Central configuration for the Tabular RAG Pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── BYOK: Sarvam AI ──
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
SARVAM_MODEL_CHAIN = [
    "sarvam-105b",
    "sarvam-30b",
]

# Provider selection: "sarvam", "openrouter", or auto-detect
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower().strip()
if not LLM_PROVIDER:
    # Auto-detect: prefer Sarvam if its key is set, else fall back to OpenRouter
    if SARVAM_API_KEY:
        LLM_PROVIDER = "sarvam"
    elif OPENROUTER_API_KEY:
        LLM_PROVIDER = "openrouter"
    else:
        LLM_PROVIDER = "openrouter"  # will raise at runtime

# Model fallback chain (free models — diverse providers to avoid shared rate limits)
MODEL_CHAIN = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "z-ai/glm-4.5-air:free",
    "openai/gpt-oss-120b:free",
    "minimax/minimax-m2.5:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
]

# ── Guardrail Settings ──
MAX_PROMPT_LENGTH = 500
MAX_TOKENS_INPUT = 4000
MAX_TOKENS_OUTPUT = 2000
MAX_QUERY_HISTORY = 5
LLM_TIMEOUT_SECONDS = 15
CIRCUIT_BREAKER_THRESHOLD = 5

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
