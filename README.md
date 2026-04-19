<div align="center">

# Tabular RAG Pipeline

### AI-Powered Financial Transaction Intelligence Platform

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Plotly](https://img.shields.io/badge/Plotly-6.7+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Sarvam AI](https://img.shields.io/badge/Sarvam_AI-105B-FF6B35?style=for-the-badge)](https://sarvam.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*A production-grade Retrieval-Augmented Generation pipeline that transforms raw financial transaction data into actionable intelligence through natural language conversation, real-time chart generation, and LLM-powered analysis — protected by multi-layer guardrails.*

[Live Demo](#quick-start) · [Architecture](#architecture) · [Features](#features) · [API Reference](#api-reference)

</div>

---

## What It Does

You ask questions in plain English. The pipeline fetches your transaction data, reasons over it with an LLM, generates publication-quality charts, and responds like a sharp financial analyst — not a chatbot.

```
"Compare my income and expenses over the last 6 months"
"Where do I spend the most money?"
"Am I saving enough? Show me a chart."
"Break down my food spending by merchant"
```

**30,000 transactions · 30 users · 27 categories · 169 merchants · 12 chart types · Real-time streaming**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (Dark UI)                          │
│               Tailwind CSS · SSE Streaming · Per-User Chat          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /api/query/stream
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1: Input & Validation                                         │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ User ID      │  │ Input Guardrails │  │ Prompt Injection      │  │
│  │ Validation   │→ │ (16 regex rules) │→ │ Cross-User Leakage    │  │
│  │              │  │                  │  │ Scope Enforcement     │  │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 2: Context Assembly                                           │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ KV Cache     │  │ Financial Profile│  │ System Prompt         │  │
│  │ (4 keys/user)│→ │ (income, savings │→ │ (Finley Persona +     │  │
│  │              │  │  rate, top cats) │  │  benchmarks + tools)  │  │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 3: LLM Reasoning + Tool Dispatch                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ Sarvam AI    │  │ Tool Calling     │  │ 12 Plotly Chart       │  │
│  │ (105B/30B)   │→ │ (JSON function   │→ │ Functions             │  │
│  │ + fallback   │  │  calls)          │  │ (dark theme, hi-DPI)  │  │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 4: Response Composition                                       │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ Output       │  │ Chart            │  │ Structured JSON       │  │
│  │ Guardrails   │→ │ Explanations     │→ │ + SSE Stream          │  │
│  │ (halluc/tox) │  │ (data-driven)    │  │ + Cache Update        │  │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | What Happens | Key Components |
|:-----:|:-------------|:---------------|
| **1** | Validate user, run 16 input guardrail checks, filter DataFrame | `guardrails.py`, `pipeline.py` |
| **2** | Build/load cached financial profile, assemble Finley system prompt with benchmarks | `cache.py`, `pipeline.py` |
| **3** | LLM generates analysis + tool calls → execute chart functions | `llm_client.py`, `visualizations.py` |
| **4** | Hallucination check, chart explanations, cache update, stream response | `guardrails.py`, `pipeline.py` |

---

## Features

### Finley — Your AI Financial Analyst

The pipeline doesn't use a generic chatbot. It embeds **Finley**, a purpose-built financial analyst persona that:

- **Leads with insight** — every response starts with the most useful finding, not filler
- **Thinks in benchmarks** — 50/30/20 rule, 20% savings target, spending spike detection (20%+ above average)
- **Speaks plainly** — direct, warm, never robotic or preachy
- **Uses friendly names** — 27 category mappings convert `RENT_HOUSING` → "Rent & Housing" automatically
- **Generates charts proactively** — calls the right visualization tool without being asked

### BYOK — Bring Your Own Key

The LLM layer supports multiple providers with automatic detection:

| Provider | Models | Status |
|----------|--------|--------|
| **Sarvam AI** (preferred) | `sarvam-105b`, `sarvam-30b` | Full support: chat, tool calling, streaming |
| **OpenRouter** (fallback) | 6 free models (Nemotron, GLM-4.5, GPT-OSS, etc.) | Full support with model chain fallback |

Set your API key in `.env` and the pipeline auto-detects the provider. No code changes needed.

### 12 Publication-Quality Chart Types

All charts use a **Linear-inspired dark theme** with Plotly + Kaleido, exported as high-DPI PNG:

| Chart | Function | Best For |
|:------|:---------|:---------|
| 📈 Spending Trend | `plot_monthly_spending_trend` | Month-over-month changes with rolling average |
| 🍩 Category Breakdown | `plot_category_breakdown` | Where your money goes (donut with total center) |
| 📊 Income vs Expense | `plot_income_vs_expense` | Savings analysis with net-savings line |
| 🏪 Top Merchants | `plot_top_merchants` | Ranked merchant spend (horizontal bars) |
| 📅 Weekly Pattern | `plot_weekly_pattern` | Average daily spend by weekday |
| 📚 Category Trends | `plot_category_trends` | Stacked area of shifting category mix |
| 📉 Expense Distribution | `plot_expense_distribution` | Transaction size histogram |
| 💰 Cumulative Flow | `plot_cumulative_flow` | Net wealth trajectory over time |
| 🔥 Spending Heatmap | `plot_monthly_heatmap` | Category × month intensity grid |
| 🎯 Savings Ratio | `plot_savings_ratio` | Monthly savings rate vs 20% target |
| 🔍 Transaction Scatter | `plot_transaction_scatter` | Every transaction by date & amount |
| 🏷️ Category Deep-Dive | `plot_category_deepdive` | Single category broken down by merchant |

### Multi-Layer Guardrails

```
Input Layer                    Output Layer                 Operational Layer
├─ Prompt injection (16 rules) ├─ Hallucination detection   ├─ Circuit breaker (5 fails → open)
├─ Cross-user leakage block    ├─ Toxicity filter           ├─ Token budget enforcement
├─ Scope enforcement           ├─ Low confidence flagging   ├─ Audit logging (hashed IDs)
└─ Length truncation (500ch)   └─ Data cross-reference      └─ 15s timeout per LLM call
```

### Intelligent Fallback System

When the LLM is unreachable, the pipeline doesn't break — it activates a **30+ intent pattern fallback engine** that:

- Detects question intent (trends, categories, merchants, savings, comparisons, budgeting, etc.)
- Generates data-driven narrative responses directly from the DataFrame
- Produces relevant charts automatically
- Handles follow-up context ("tell me more", "explain that", "show a better chart")
- Provides conversation-aware catch-all responses

### Real-Time Streaming

The `/api/query/stream` endpoint uses **Server-Sent Events** to stream LLM tokens in real-time:

```
event: token    → individual response tokens
event: chart    → base64-encoded chart PNG
event: done     → final metadata (latency, cache hit, guardrail flags)
```

### Premium Dark UI

A single-page app built with **Tailwind CSS** featuring:
- Inter + JetBrains Mono typography
- Animated gradient background with floating blobs
- Per-user chat persistence via localStorage
- Inline chart rendering with zoom support
- User selector dropdown with transaction counts
- Responsive design for all screen sizes

---

## Quick Start

### Prerequisites

- Python 3.12+
- A Sarvam AI or OpenRouter API key

### Installation

```bash
# Clone the repository
git clone https://github.com/pratiksinha312511/Tabular_RAG_pipeline.git
cd Tabular_RAG_pipeline

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install plotly kaleido     # For chart generation
```

### Configuration

Create a `.env` file in the project root:

```env
# Choose one (or both — Sarvam takes priority):
SARVAM_API_KEY=your_sarvam_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Data file (default: data/transactions.csv)
CSV_FILE=data/transactionsmore.csv

# Optional: force a specific provider
# LLM_PROVIDER=sarvam
```

### Run

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## API Reference

| Method | Endpoint | Description |
|:------:|:---------|:------------|
| `GET` | `/` | Serve the frontend UI |
| `GET` | `/api/users` | List all users with transaction counts |
| `POST` | `/api/query` | Synchronous query — full response at once |
| `POST` | `/api/query/stream` | SSE streaming — tokens + charts in real-time |
| `POST` | `/api/conversation/clear` | Clear a user's conversation history |

### Request Body (POST endpoints)

```json
{
  "user_id": "usr_a1b2c3d4",
  "prompt": "What did I spend the most on last month?"
}
```

### Response Structure

```json
{
  "user_name": "Jose BazBaz",
  "response": "Your biggest spending category is Rent & Housing at $5,550.00...",
  "data_summary": {
    "total_expenses": 24510.50,
    "total_income": 31200.00,
    "num_transactions": 347,
    "monthly_expenses": { "2025-01-31": 4102.50 },
    "category_totals": { "Rent & Housing": 5550.00 }
  },
  "visualizations": ["data:image/png;base64,..."],
  "cache_hit": true,
  "latency_ms": 1240,
  "guardrail_flags": []
}
```

---

## Programmatic Usage

```python
import pandas as pd
from pipeline import TransactionRAGPipeline

df = pd.read_csv("data/transactionsmore.csv")
pipeline = TransactionRAGPipeline(df=df)

# Synchronous
result = pipeline.run(user_id="usr_a1b2c3d4", prompt="Am I saving enough?")
print(result["response"])

# Streaming (async generator)
async for chunk in pipeline.run_stream(user_id="usr_a1b2c3d4", prompt="Show my trends"):
    print(chunk, end="", flush=True)
```

---

## Project Structure

```
Tabular_RAG_pipeline/
├── server.py              # FastAPI app — endpoints, SSE streaming, static files
├── pipeline.py            # 4-stage RAG orchestrator — Finley persona, 30+ intents
├── llm_client.py          # BYOK LLM client — Sarvam AI + OpenRouter, streaming
├── config.py              # Central config — models, timeouts, provider detection
├── guardrails.py          # Input/output/operational guardrails + audit logging
├── cache.py               # KV cache — profiles, query history, viz state
├── visualizations.py      # 12 Plotly chart functions + TOOL_SCHEMAS for LLM
├── categories.py          # 27 friendly category name mappings
├── static/
│   └── index.html         # Premium dark UI — Tailwind, SSE, per-user persistence
├── data/
│   └── transactionsmore.csv   # 30K transactions, 30 users, 27 categories
├── tests/
│   ├── test_comprehensive.py  # Full test suite
│   └── test_quick.py          # Smoke tests
├── requirements.txt
├── .env                   # API keys (gitignored)
└── .gitignore
```

---

## Data Model

The pipeline operates on a transaction DataFrame with this schema:

| Column | Type | Description |
|:-------|:-----|:------------|
| `user_id` | string | Unique user identifier |
| `user_name` | string | Display name |
| `transaction_date` | datetime | Transaction timestamp |
| `transaction_amount` | float | Positive = expense, Negative = income |
| `transaction_category_detail` | string | Raw category (e.g. `RENT_HOUSING`) |
| `merchant_name` | string | Merchant or source name |

**Dataset:** 30,000 transactions across 30 users, 27 spending/income categories, 169 unique merchants, spanning Jan 2024 – Sep 2025.

---

## KV Cache Strategy

Each user gets 4 cache keys to avoid redundant computation:

| Key | Contents | TTL |
|:----|:---------|:----|
| `user:{id}:profile` | Name, date range, top categories, avg monthly income/spend, savings rate | Session |
| `user:{id}:query_history` | Last 5 (prompt, operation, result_summary) tuples | Session |
| `user:{id}:viz_state` | Last chart type, parameters, filters | Session |
| `user:{id}:data_summary` | Aggregated financial summary | Session |

---

## Error Handling

| Scenario | Behavior |
|:---------|:---------|
| LLM unreachable | Activates 30+ intent fallback engine with data-driven responses |
| All models fail | Circuit breaker opens for 60s, serves cached/DataFrame responses |
| Invalid user ID | Structured error with suggestion to check available users |
| Empty query results | Explains why + suggests alternative queries |
| Malformed LLM output | Retry with next model in chain, then graceful fallback |
| Sarvam `null` tool_calls | Handled — `(delta.get("tool_calls") or [])` |
| Empty streaming chunks | Handled — skips chunks with no choices |

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **LLM** | Sarvam AI (sarvam-105b / 30b), OpenRouter (6 model fallback chain) |
| **Data** | Pandas, NumPy |
| **Visualization** | Plotly 6.7, Kaleido (server-side PNG) |
| **Frontend** | Vanilla JS, Tailwind CSS, SSE (EventSource) |
| **Security** | Input/output guardrails, circuit breaker, audit logging, hashed PIIs |

---

## License

MIT

---

<div align="center">

Built with precision for the **CampusLearning Developer Assessment**

*Tabular RAG Pipeline — turning transaction data into financial intelligence.*

</div>

## Data

3 users × ~100-120 transactions each (May–Dec 2025):
- `usr_a1b2c3d4` — Jose BazBaz
- `usr_e5f6g7h8` — Sarah Collins
- `usr_i9j0k1l2` — Marcus Johnson

## Tech Stack

- **Python 3.12**, Pandas, Matplotlib
- **OpenRouter** (free-tier LLM with tool calling)
- **python-dotenv** for config management
