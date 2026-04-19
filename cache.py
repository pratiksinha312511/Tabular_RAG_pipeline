"""User-specific KV cache layer for the RAG pipeline."""

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from config import MAX_QUERY_HISTORY


class KVCache:
    """In-memory KV cache with user-specific namespacing.

    Key patterns
    ------------
    user:{id}:profile        – name, date range, top categories, avg spend
    user:{id}:query_history  – last N (prompt, operation, result_summary)
    user:{id}:viz_state      – last chart type, axes, filters
    """

    def __init__(self, max_query_history: int = MAX_QUERY_HISTORY):
        self._store: dict[str, Any] = {}
        self.max_query_history = max_query_history

    # ── generic get / set ──

    def _key(self, user_id: str, namespace: str) -> str:
        return f"user:{user_id}:{namespace}"

    def get(self, user_id: str, namespace: str) -> Optional[Any]:
        return self._store.get(self._key(user_id, namespace))

    def set(self, user_id: str, namespace: str, value: Any) -> None:
        self._store[self._key(user_id, namespace)] = value

    # ── profile ──

    def get_profile(self, user_id: str) -> Optional[dict]:
        return self.get(user_id, "profile")

    def set_profile(self, user_id: str, profile: dict) -> None:
        self.set(user_id, "profile", profile)

    def build_profile(self, user_id: str, user_df: pd.DataFrame) -> dict:
        """Compute user profile from their transaction rows and cache it."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]

        profile = {
            "user_name": user_df["user_name"].iloc[0],
            "date_range": {
                "start": str(user_df["transaction_date"].min().date()),
                "end": str(user_df["transaction_date"].max().date()),
            },
            "total_transactions": len(user_df),
            "top_expense_categories": (
                expenses.groupby("transaction_category_detail")["transaction_amount"]
                .sum()
                .nlargest(5)
                .to_dict()
            ),
            "avg_monthly_spend": round(
                expenses.set_index("transaction_date")
                .resample("ME")["transaction_amount"]
                .sum()
                .mean(),
                2,
            ) if not expenses.empty else 0,
            "avg_monthly_income": round(
                abs(
                    income.set_index("transaction_date")
                    .resample("ME")["transaction_amount"]
                    .sum()
                    .mean()
                ),
                2,
            ) if not income.empty else 0,
            "cached_at": datetime.now().isoformat(),
        }
        self.set_profile(user_id, profile)
        return profile

    # ── query history ──

    def get_query_history(self, user_id: str) -> list[dict]:
        return self.get(user_id, "query_history") or []

    def add_query(
        self,
        user_id: str,
        prompt: str,
        operation: str,
        result_summary: str,
    ) -> None:
        history = self.get_query_history(user_id)
        history.append({
            "prompt": prompt,
            "operation": operation,
            "result_summary": result_summary,
            "timestamp": datetime.now().isoformat(),
        })
        if len(history) > self.max_query_history:
            history = history[-self.max_query_history :]
        self.set(user_id, "query_history", history)

    # ── visualization state ──

    def get_viz_state(self, user_id: str) -> Optional[dict]:
        return self.get(user_id, "viz_state")

    def set_viz_state(
        self, user_id: str, chart_type: str, axes: dict, filters: dict
    ) -> None:
        self.set(user_id, "viz_state", {
            "last_chart_type": chart_type,
            "axes": axes,
            "filters": filters,
            "updated_at": datetime.now().isoformat(),
        })

    # ── conversation history (full messages for multi-turn) ──

    def get_conversation(self, user_id: str) -> list[dict]:
        """Return the recent conversation messages [{role, content}, ...]."""
        return self.get(user_id, "conversation") or []

    def add_conversation_turn(self, user_id: str, role: str, content: str, max_turns: int = 10) -> None:
        """Append a message to the conversation history, keeping last N turns."""
        conv = self.get_conversation(user_id)
        conv.append({"role": role, "content": content})
        # Keep last max_turns pairs (user+assistant = 2 messages each)
        max_messages = max_turns * 2
        if len(conv) > max_messages:
            conv = conv[-max_messages:]
        self.set(user_id, "conversation", conv)

    def clear_conversation(self, user_id: str) -> None:
        self.set(user_id, "conversation", [])
