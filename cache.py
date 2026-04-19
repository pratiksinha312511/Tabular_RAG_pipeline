"""User-specific KV cache layer for the RAG pipeline."""

import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from config import MAX_QUERY_HISTORY

# Cache TTL and eviction settings
CACHE_TTL_SECONDS = 1800      # 30 minutes — profiles rebuild on miss (same as first request)
MAX_CACHED_USERS = 100        # Max unique users in cache before LRU eviction


class KVCache:
    """In-memory KV cache with user-specific namespacing, TTL, and LRU eviction.

    Key patterns
    ------------
    user:{id}:profile        – name, date range, top categories, avg spend
    user:{id}:query_history  – last N (prompt, operation, result_summary)
    user:{id}:viz_state      – last chart type, axes, filters
    user:{id}:conversation   – multi-turn conversation messages

    TTL: entries expire after CACHE_TTL_SECONDS (30 min).
    Eviction: when MAX_CACHED_USERS is exceeded, the least-recently-accessed user is evicted.
    """

    def __init__(self, max_query_history: int = MAX_QUERY_HISTORY):
        self._store: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}   # key → last access time
        self._user_access: dict[str, float] = {}   # user_id → last access time (for LRU)
        self.max_query_history = max_query_history

    # ── generic get / set ──

    def _key(self, user_id: str, namespace: str) -> str:
        return f"user:{user_id}:{namespace}"

    def _touch_user(self, user_id: str) -> None:
        """Update the last-access timestamp for LRU tracking."""
        self._user_access[user_id] = time.time()

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has exceeded its TTL."""
        ts = self._timestamps.get(key)
        if ts is None:
            return True
        return (time.time() - ts) > CACHE_TTL_SECONDS

    def _evict_if_needed(self) -> None:
        """If we exceed MAX_CACHED_USERS, evict the least-recently-used user."""
        if len(self._user_access) <= MAX_CACHED_USERS:
            return
        # Find the user with the oldest access time
        oldest_uid = min(self._user_access, key=self._user_access.get)
        self._evict_user(oldest_uid)

    def _evict_user(self, user_id: str) -> None:
        """Remove all cached data for a specific user."""
        prefix = f"user:{user_id}:"
        keys_to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_delete:
            self._store.pop(k, None)
            self._timestamps.pop(k, None)
        self._user_access.pop(user_id, None)

    def get(self, user_id: str, namespace: str) -> Optional[Any]:
        key = self._key(user_id, namespace)
        if self._is_expired(key):
            self._store.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        self._touch_user(user_id)
        return self._store.get(key)

    def set(self, user_id: str, namespace: str, value: Any) -> None:
        self._evict_if_needed()
        key = self._key(user_id, namespace)
        self._store[key] = value
        self._timestamps[key] = time.time()
        self._touch_user(user_id)

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
