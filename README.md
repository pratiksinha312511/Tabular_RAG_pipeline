# Tabular RAG Pipeline

A production-grade AI pipeline that takes a `user_id` & natural language prompt, fetches that user's financial transactions from a pre-loaded Pandas DataFrame, generates a tailored analytical response via LLM, and produces contextual visualizations through tool calling — all accelerated by a user-specific KV cache layer and protected by LLM guardrails.

## Architecture

```
User Prompt ──► Input Guardrails ──► Context Assembly ──► LLM (OpenRouter) ──► Response
                  │                     │                    │  ▲                  │
                  │                     ▼                    │  │                  ▼
                  │                 KV Cache                 │  │            Output Guardrails
                  │               (profile,                  ▼  │                  │
                  │              history, viz)          Tool Calls ──► Charts      │
                  ▼                                                               ▼
              Reject / Redirect                                           Structured JSON
```

### 4-Stage Pipeline

| Stage | Description |
|-------|-------------|
| **1. Input & User Data Fetch** | Validate `user_id`, filter DataFrame, load/build cached profile |
| **2. Context Assembly** | Build LLM prompt with user profile, column descriptions, few-shot history |
| **3. LLM Reasoning + Tool Dispatch** | Call OpenRouter LLM with tool schemas; parse & execute tool calls |
| **4. Response Composition** | Combine text + charts, run output guardrails, update cache, return JSON |

## Setup

```bash
# 1. Clone
git clone https://github.com/pratiksinha312511/Tabular_RAG_pipeline.git
cd Tabular_RAG_pipeline

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
# Create a .env file with:
OPENROUTER_API_KEY=your_key_here

# 5. Reconstruct data (if needed)
python scripts/reconstruct_data.py

# 6. Run
python main.py
```

## Usage

```python
import pandas as pd
from pipeline import TransactionRAGPipeline

df = pd.read_csv("data/transactions.csv")
pipeline = TransactionRAGPipeline(df=df)

result = pipeline.run(
    user_id="usr_a1b2c3d4",
    prompt="What did I spend the most on last month?"
)

# Output structure
# {
#   "user_name": "Jose BazBaz",
#   "response": "Your top category was RENT_HOUSING at $1,850...",
#   "data_summary": { ... },
#   "visualizations": ["./output/jose_bazbaz_category_breakdown_20260418.png"],
#   "cache_hit": true,
#   "latency_ms": 820,
#   "guardrail_flags": []
# }
```

## Features

### KV Cache (user-specific)
- `user:{id}:profile` — name, date range, top categories, avg monthly spend
- `user:{id}:query_history` — last N (prompt, operation, result_summary) tuples
- `user:{id}:viz_state` — last chart type, axes, filters

### Visualizations (LLM tool calls)
- **`plot_monthly_spending_trend`** — line chart + rolling average
- **`plot_category_breakdown`** — donut chart with total in centre
- **`plot_income_vs_expense`** — grouped bars + net savings line

### Guardrails
- **Input:** prompt injection detection, scope enforcement, length limiting, cross-user leakage prevention
- **Output:** hallucination check (cross-ref vs DataFrame), confidence gating
- **Operational:** token budget, audit logging, timeout & circuit breaker

### Error Handling
- LLM unreachable → fallback to cached data / raw DataFrame stats
- Invalid user → structured error response
- Empty results → explains why + suggests alternatives
- Malformed LLM output → retry then graceful fallback

## Data

3 users × ~100-120 transactions each (May–Dec 2025):
- `usr_a1b2c3d4` — Jose BazBaz
- `usr_e5f6g7h8` — Sarah Collins
- `usr_i9j0k1l2` — Marcus Johnson

## Tech Stack

- **Python 3.12**, Pandas, Matplotlib
- **OpenRouter** (free-tier LLM with tool calling)
- **python-dotenv** for config management
