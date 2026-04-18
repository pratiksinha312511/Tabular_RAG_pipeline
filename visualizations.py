"""Visualization tool functions registered as LLM tool-call targets."""

import os
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config import OUTPUT_DIR

# ═══════════════════════════════════════════════════════════════════
#  JSON TOOL SCHEMAS  (OpenAI-compatible format for OpenRouter)
# ═══════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "plot_monthly_spending_trend",
            "description": (
                "Plot a line chart showing monthly spending totals with a "
                "rolling average overlay. Use when the user asks about spending "
                "trends over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {
                        "type": "integer",
                        "description": "Lookback period in months (default 6)",
                    },
                    "category_filter": {
                        "type": "string",
                        "description": "Optional category to filter (e.g. FOOD, HOUSING)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_category_breakdown",
            "description": (
                "Plot a donut chart showing spending split by category with "
                "total spend in the centre. Use when the user asks where their "
                "money is going."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "period": {
                        "type": "string",
                        "description": "Time window: last_month, last_3_months, last_6_months, all_time",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Top N categories to show; rest grouped as Other (default 7)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_income_vs_expense",
            "description": (
                "Plot grouped bars (green income, red expense) with an optional "
                "net-savings line. Use when the user asks if they are saving money "
                "or about income vs expenses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {
                        "type": "integer",
                        "description": "Lookback period in months (default 6)",
                    },
                    "show_net_line": {
                        "type": "boolean",
                        "description": "Overlay net savings line (default true)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _user_df(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    return df[df["user_id"] == user_id].copy()


def _filter_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    cutoff = df["transaction_date"].max() - pd.DateOffset(months=months)
    return df[df["transaction_date"] >= cutoff]


def _period_to_months(period: str) -> int:
    return {"last_month": 1, "last_3_months": 3, "last_6_months": 6, "all_time": 120}.get(period, 3)


def _save(fig, user_name: str, tag: str) -> str:
    clean = user_name.replace(" ", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{clean}_{tag}_{ts}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════════
#  1.  Monthly Spending Trend
# ═══════════════════════════════════════════════════════════════════

def plot_monthly_spending_trend(
    df: pd.DataFrame,
    user_id: str,
    months: int = 6,
    category_filter: Optional[str] = None,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0]

    if category_filter:
        expenses = expenses[
            expenses["transaction_category_detail"].str.contains(
                category_filter, case=False, na=False
            )
        ]
    if expenses.empty:
        return ""

    monthly = (
        expenses.set_index("transaction_date")
        .resample("ME")["transaction_amount"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly.index, monthly.values, marker="o", lw=2, color="#2196F3", label="Monthly Spending")

    if len(monthly) >= 3:
        rolling = monthly.rolling(window=3, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values, ls="--", lw=2, color="#FF9800", label="3-Mo Rolling Avg")

    ax.set_title(f"Monthly Spending Trend — {udf['user_name'].iloc[0]}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    return _save(fig, udf["user_name"].iloc[0], "spending_trend")

# ═══════════════════════════════════════════════════════════════════
#  2.  Category Breakdown (Donut)
# ═══════════════════════════════════════════════════════════════════

def plot_category_breakdown(
    df: pd.DataFrame,
    user_id: str,
    period: str = "last_3_months",
    top_n: int = 7,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, _period_to_months(period))
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    cat_all = expenses.groupby("transaction_category_detail")["transaction_amount"].sum()
    cat_top = cat_all.nlargest(top_n)
    other = cat_all.sum() - cat_top.sum()
    if other > 0:
        cat_top["Other"] = other
    total = cat_top.sum()

    labels = [c.replace("_", " > ", 1).replace("_", " ").title() for c in cat_top.index]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_top)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(
        cat_top.values, labels=labels, autopct="%1.1f%%",
        colors=colors, pctdistance=0.8, startangle=90,
        wedgeprops={"width": 0.4},
    )
    ax.text(0, 0, f"${total:,.0f}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set_title(f"Spending by Category — {udf['user_name'].iloc[0]}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, udf["user_name"].iloc[0], "category_breakdown")

# ═══════════════════════════════════════════════════════════════════
#  3.  Income vs Expense
# ═══════════════════════════════════════════════════════════════════

def plot_income_vs_expense(
    df: pd.DataFrame,
    user_id: str,
    months: int = 6,
    show_net_line: bool = True,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    if udf.empty:
        return ""

    monthly_exp = (
        udf[udf["transaction_amount"] > 0]
        .set_index("transaction_date")
        .resample("ME")["transaction_amount"].sum()
    )
    monthly_inc = (
        udf[udf["transaction_amount"] < 0]
        .set_index("transaction_date")
        .resample("ME")["transaction_amount"].sum().abs()
    )

    idx = monthly_exp.index.union(monthly_inc.index)
    monthly_exp = monthly_exp.reindex(idx, fill_value=0)
    monthly_inc = monthly_inc.reindex(idx, fill_value=0)

    x = np.arange(len(idx))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, monthly_inc.values, w, label="Income", color="#4CAF50", alpha=0.85)
    ax.bar(x + w / 2, monthly_exp.values, w, label="Expenses", color="#F44336", alpha=0.85)

    if show_net_line:
        net = monthly_inc.values - monthly_exp.values
        ax2 = ax.twinx()
        ax2.plot(x, net, color="#FF9800", marker="D", lw=2, label="Net Savings")
        ax2.axhline(0, color="gray", ls="--", alpha=0.5)
        ax2.set_ylabel("Net Savings ($)")
        ax2.legend(loc="upper left")

    ax.set_title(f"Income vs Expenses — {udf['user_name'].iloc[0]}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount ($)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b %Y") for d in idx], rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save(fig, udf["user_name"].iloc[0], "income_vs_expense")

# ═══════════════════════════════════════════════════════════════════
#  Dispatch map
# ═══════════════════════════════════════════════════════════════════

VIZ_FUNCTIONS = {
    "plot_monthly_spending_trend": plot_monthly_spending_trend,
    "plot_category_breakdown": plot_category_breakdown,
    "plot_income_vs_expense": plot_income_vs_expense,
}
