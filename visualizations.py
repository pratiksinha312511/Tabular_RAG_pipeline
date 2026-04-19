"""Visualization tool functions registered as LLM tool-call targets.

Uses Plotly + Kaleido for premium, publication-quality dark-themed charts.
"""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import OUTPUT_DIR
from categories import friendly_category

# ═══════════════════════════════════════════════════════════════════
#  Premium Dark Theme — matches the Linear-inspired frontend
# ═══════════════════════════════════════════════════════════════════

_BG        = "#0D0F11"   # deep background
_PAPER     = "#141619"   # card / paper
_GRID      = "#1E2228"   # subtle grid lines
_TEXT      = "#E2E4E9"   # primary text
_TEXT_SEC  = "#8B8F96"   # secondary / muted text
_ACCENT    = "#6C5CE7"   # accent purple (matches UI --accent)
_GREEN     = "#00D68F"   # income / positive
_RED       = "#FF6B6B"   # expense / negative
_ORANGE    = "#FFA502"   # net / secondary highlight
_BLUE      = "#4ECDC4"   # teal-blue accent
_CYAN      = "#45B7D1"   # info / rolling avg

# Premium colour palette for categories / series
_PALETTE = [
    "#7C3AED", "#00D4AA", "#FF6B6B", "#FFA502", "#4ECDC4",
    "#45B7D1", "#A78BFA", "#F472B6", "#FBBF24", "#34D399",
    "#818CF8", "#FB923C", "#38BDF8", "#E879F9", "#2DD4BF",
]

# Gradient palette for histograms / heatmaps
_GRADIENT = ["#00D4AA", "#4ECDC4", "#45B7D1", "#7C3AED", "#A78BFA", "#F472B6", "#FF6B6B"]

def _base_layout(title: str, **kwargs) -> dict:
    """Return a shared Plotly layout dict for the premium dark theme."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=_PAPER,
        plot_bgcolor=_BG,
        font=dict(family="Inter, SF Pro Display, -apple-system, sans-serif", color=_TEXT, size=13),
        title=dict(
            text=title,
            font=dict(size=18, color=_TEXT, family="Inter, SF Pro Display, sans-serif"),
            x=0.5, xanchor="center", y=0.97,
        ),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(
            gridcolor=_GRID, gridwidth=1, zeroline=False,
            tickfont=dict(color=_TEXT_SEC, size=11),
            title_font=dict(color=_TEXT_SEC, size=12),
        ),
        yaxis=dict(
            gridcolor=_GRID, gridwidth=1, zeroline=False,
            tickfont=dict(color=_TEXT_SEC, size=11),
            title_font=dict(color=_TEXT_SEC, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(color=_TEXT_SEC, size=11),
        ),
        hoverlabel=dict(
            bgcolor=_PAPER, font_size=12, font_color=_TEXT,
            bordercolor=_GRID,
        ),
    )
    layout.update(kwargs)
    return layout

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
    {
        "type": "function",
        "function": {
            "name": "plot_top_merchants",
            "description": (
                "Plot a horizontal bar chart of top merchants by total spend. "
                "Use when the user asks about where they shop or top merchants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "top_n": {
                        "type": "integer",
                        "description": "Number of merchants to show (default 10)",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Lookback period in months (default 12)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_weekly_pattern",
            "description": (
                "Plot a bar chart showing average spending by day of the week. "
                "Use when the user asks about spending patterns, which days they spend most, "
                "or weekly habits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {
                        "type": "integer",
                        "description": "Lookback period in months (default 6)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_category_trends",
            "description": (
                "Plot a stacked area chart showing how spending across top categories "
                "changes month over month. Use when the user wants to compare categories "
                "over time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {
                        "type": "integer",
                        "description": "Lookback period in months (default 6)",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top categories to show (default 5)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_expense_distribution",
            "description": (
                "Plot a histogram showing the distribution of transaction amounts. "
                "Use when the user asks about transaction sizes, small vs large purchases, "
                "or spending distribution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "bins": {
                        "type": "integer",
                        "description": "Number of histogram bins (default 20)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_cumulative_flow",
            "description": (
                "Plot cumulative income vs cumulative expenses over time with a "
                "shaded gap showing net savings. Use when the user asks about financial "
                "trajectory, overall progress, or cumulative totals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {
                        "type": "integer",
                        "description": "Lookback period in months (default 12)",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_monthly_heatmap",
            "description": (
                "Plot a heatmap grid of categories × months showing spending intensity. "
                "Use when the user asks for a heatmap, calendar view, or category-month grid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {"type": "integer", "description": "Lookback period in months (default 6)"},
                    "top_n": {"type": "integer", "description": "Number of top categories (default 8)"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_savings_ratio",
            "description": (
                "Plot monthly savings rate as a bar chart with a 20% target line. "
                "Use when the user asks about savings rate, savings ratio, or how much they save each month."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {"type": "integer", "description": "Lookback period in months (default 8)"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_transaction_scatter",
            "description": (
                "Plot each transaction as a dot on a date × amount scatter chart, colour-coded "
                "by category. Use when the user asks for a timeline, scatter plot, or to see all transactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "months": {"type": "integer", "description": "Lookback period in months (default 6)"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_category_deepdive",
            "description": (
                "Plot a horizontal bar chart breaking down a single category by merchant. "
                "Use when the user asks to drill down into a specific category, or wants "
                "merchant details within a category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Target user ID"},
                    "category": {"type": "string", "description": "Category to deep-dive into (default: top category)"},
                    "months": {"type": "integer", "description": "Lookback period in months (default 12)"},
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
    fixed = {"last_month": 1, "last_3_months": 3, "last_6_months": 6, "all_time": 120}
    if period in fixed:
        return fixed[period]
    # Dynamic: "last_N_months" → N
    import re
    m = re.match(r"last_(\d+)_months?", period)
    if m:
        return int(m.group(1))
    return 3  # default


def _save(fig: go.Figure, user_name: str, tag: str) -> str:
    """Save a Plotly figure as a high-DPI PNG."""
    clean = user_name.replace(" ", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{clean}_{tag}_{ts}.png")
    fig.write_image(path, width=1100, height=650, scale=2, engine="kaleido")
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
    user_name = udf["user_name"].iloc[0]

    fig = go.Figure()

    # Gradient area fill under the line
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.values,
        mode="lines+markers",
        name="Monthly Spending",
        line=dict(color=_ACCENT, width=3, shape="spline"),
        marker=dict(size=10, color=_ACCENT, line=dict(width=2, color=_PAPER)),
        fill="tozeroy",
        fillcolor="rgba(108,92,231,0.15)",
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Rolling average
    if len(monthly) >= 3:
        rolling = monthly.rolling(window=3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling.values,
            mode="lines",
            name="3-Mo Rolling Avg",
            line=dict(color=_CYAN, width=2, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Avg: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        f"Monthly Spending Trend — {user_name}",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        xaxis_dtick="M1",
        xaxis_tickformat="%b %Y",
    ))

    return _save(fig, user_name, "spending_trend")

# ═══════════════════════════════════════════════════════════════════
#  2.  Category Breakdown (Donut)
# ═══════════════════════════════════════════════════════════════════

def plot_category_breakdown(
    df: pd.DataFrame,
    user_id: str,
    period: str = "last_3_months",
    top_n: int = 7,
    months: int | None = None,
) -> str:
    udf = _user_df(df, user_id)
    # months param overrides period string when provided
    filter_months = months if months is not None else _period_to_months(period)
    udf = _filter_months(udf, filter_months)
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    cat_all = expenses.groupby("transaction_category_detail")["transaction_amount"].sum()
    cat_top = cat_all.nlargest(top_n)
    other = cat_all.sum() - cat_top.sum()
    if other > 0:
        cat_top["Other"] = other
    total = cat_top.sum()

    labels = [friendly_category(c) if c != "Other" else "Other" for c in cat_top.index]
    colors = _PALETTE[:len(cat_top)]
    user_name = udf["user_name"].iloc[0]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=cat_top.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color=_TEXT),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f} (%{percent})<extra></extra>",
        sort=False,
    ))

    # Centre annotation with total
    fig.add_annotation(
        text=f"<b>${total:,.0f}</b><br><span style='font-size:11px;color:{_TEXT_SEC}'>Total</span>",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=22, color=_TEXT),
    )

    fig.update_layout(**_base_layout(
        f"Spending by Category — {user_name}",
        showlegend=True,
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_SEC, size=11),
        ),
    ))

    return _save(fig, user_name, "category_breakdown")

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
    user_name = udf["user_name"].iloc[0]
    month_labels = [d.strftime("%b %Y") for d in idx]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=month_labels, y=monthly_inc.values,
        name="Income",
        marker=dict(color=_GREEN, cornerradius=4),
        hovertemplate="<b>%{x}</b><br>Income: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=month_labels, y=monthly_exp.values,
        name="Expenses",
        marker=dict(color=_RED, cornerradius=4),
        hovertemplate="<b>%{x}</b><br>Expenses: $%{y:,.0f}<extra></extra>",
    ))

    if show_net_line:
        net = monthly_inc.values - monthly_exp.values
        fig.add_trace(go.Scatter(
            x=month_labels, y=net,
            mode="lines+markers",
            name="Net Savings",
            yaxis="y2",
            line=dict(color=_ORANGE, width=3),
            marker=dict(size=9, symbol="diamond", color=_ORANGE, line=dict(width=2, color=_PAPER)),
            hovertemplate="<b>%{x}</b><br>Net: $%{y:,.0f}<extra></extra>",
        ))

    layout_kwargs = dict(
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        bargap=0.25,
        bargroupgap=0.1,
    )
    if show_net_line:
        layout_kwargs["yaxis2"] = dict(
            title="Net Savings ($)",
            overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)", zeroline=True,
            zerolinecolor=_GRID, zerolinewidth=1,
            tickfont=dict(color=_ORANGE, size=11),
            title_font=dict(color=_ORANGE, size=12),
            tickprefix="$", tickformat=",",
        )

    fig.update_layout(**_base_layout(f"Income vs Expenses — {user_name}", **layout_kwargs))

    return _save(fig, user_name, "income_vs_expense")

# ═══════════════════════════════════════════════════════════════════
#  4.  Top Merchants (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════

def plot_top_merchants(
    df: pd.DataFrame,
    user_id: str,
    top_n: int = 10,
    months: int = 12,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    merch = expenses.groupby("merchant_name")["transaction_amount"].sum().nlargest(top_n)
    total = expenses["transaction_amount"].sum()
    user_name = udf["user_name"].iloc[0]

    # Reverse so highest is at top
    merch = merch.iloc[::-1]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(merch))]

    annotations = [f"${v:,.0f} ({v/total*100:.1f}%)" for v in merch.values]

    fig = go.Figure(go.Bar(
        y=merch.index,
        x=merch.values,
        orientation="h",
        marker=dict(
            color=colors,
            cornerradius=4,
            line=dict(width=0),
        ),
        text=annotations,
        textposition="outside",
        textfont=dict(color=_TEXT_SEC, size=11),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        f"Top {top_n} Merchants — {user_name}",
        xaxis_title="Total Spend ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        height=max(500, top_n * 55),
        margin=dict(l=120, r=100, t=60, b=50),
    ))

    return _save(fig, user_name, "top_merchants")

# ═══════════════════════════════════════════════════════════════════
#  5.  Spending by Day of Week
# ═══════════════════════════════════════════════════════════════════

def plot_weekly_pattern(
    df: pd.DataFrame,
    user_id: str,
    months: int = 6,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0].copy()
    if expenses.empty:
        return ""

    expenses["day_of_week"] = expenses["transaction_date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_avg = expenses.groupby("day_of_week")["transaction_amount"].mean().reindex(day_order).fillna(0)
    day_count = expenses.groupby("day_of_week").size().reindex(day_order).fillna(0)
    user_name = udf["user_name"].iloc[0]

    colors = [_GREEN if d in ("Saturday", "Sunday") else _BLUE for d in day_order]
    short_names = [d[:3] for d in day_order]

    fig = go.Figure(go.Bar(
        x=short_names,
        y=day_avg.values,
        marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
        text=[f"${v:,.0f}<br><span style='font-size:10px;color:{_TEXT_SEC}'>{int(c)} txns</span>"
              for v, c in zip(day_avg.values, day_count.values)],
        textposition="outside",
        textfont=dict(size=11, color=_TEXT),
        hovertemplate="<b>%{x}</b><br>Avg: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        f"Average Spending by Day — {user_name}",
        yaxis_title="Average Spend ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
    ))

    # Custom legend for weekday vs weekend
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=_BLUE), name="Weekday", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=_GREEN), name="Weekend", showlegend=True))

    return _save(fig, user_name, "weekly_pattern")

# ═══════════════════════════════════════════════════════════════════
#  6.  Category Trends (Stacked Area)
# ═══════════════════════════════════════════════════════════════════

def plot_category_trends(
    df: pd.DataFrame,
    user_id: str,
    months: int = 6,
    top_n: int = 5,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    pivot = (
        expenses.set_index("transaction_date")
        .groupby([pd.Grouper(freq="ME"), "transaction_category_detail"])["transaction_amount"]
        .sum()
        .unstack(fill_value=0)
    )
    if pivot.empty:
        return ""

    cat_totals = pivot.sum().nlargest(top_n)
    other_cols = [c for c in pivot.columns if c not in cat_totals.index]
    plot_df = pivot[cat_totals.index].copy()
    if other_cols:
        plot_df["Other"] = pivot[other_cols].sum(axis=1)

    plot_df.columns = [friendly_category(c) if c != "Other" else "Other" for c in plot_df.columns]
    user_name = udf["user_name"].iloc[0]

    fig = go.Figure()
    for i, col in enumerate(plot_df.columns):
        color = _PALETTE[i % len(_PALETTE)]
        # Convert hex to rgba for fill transparency
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_rgba = f"rgba({r},{g},{b},0.55)"
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[col].values,
            mode="lines",
            name=col,
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=fill_rgba,
            hovertemplate=f"<b>{col}</b><br>" + "%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        f"Category Spending Over Time — {user_name}",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        xaxis_dtick="M1",
        xaxis_tickformat="%b %Y",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
    ))

    return _save(fig, user_name, "category_trends")

# ═══════════════════════════════════════════════════════════════════
#  7.  Transaction Amount Distribution (Histogram)
# ═══════════════════════════════════════════════════════════════════

def plot_expense_distribution(
    df: pd.DataFrame,
    user_id: str,
    bins: int = 20,
) -> str:
    udf = _user_df(df, user_id)
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    amounts = expenses["transaction_amount"].values
    user_name = udf["user_name"].iloc[0]
    median_val = float(np.median(amounts))
    mean_val = float(np.mean(amounts))

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=amounts,
        nbinsx=bins,
        marker=dict(
            color=_ACCENT,
            line=dict(color=_PAPER, width=1),
        ),
        opacity=0.85,
        hovertemplate="Range: $%{x}<br>Count: %{y}<extra></extra>",
    ))

    # Median line
    fig.add_vline(
        x=median_val, line=dict(color=_RED, width=2, dash="dash"),
        annotation_text=f"Median ${median_val:,.0f}",
        annotation_font=dict(color=_RED, size=11),
        annotation_position="top right",
    )
    # Mean line
    fig.add_vline(
        x=mean_val, line=dict(color=_CYAN, width=2, dash="dash"),
        annotation_text=f"Mean ${mean_val:,.0f}",
        annotation_font=dict(color=_CYAN, size=11),
        annotation_position="top left",
    )

    fig.update_layout(**_base_layout(
        f"Transaction Amount Distribution — {user_name}",
        xaxis_title="Transaction Amount ($)",
        yaxis_title="Number of Transactions",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        bargap=0.05,
    ))

    return _save(fig, user_name, "expense_distribution")

# ═══════════════════════════════════════════════════════════════════
#  8.  Cumulative Income vs Expenses (Area)
# ═══════════════════════════════════════════════════════════════════

def plot_cumulative_flow(
    df: pd.DataFrame,
    user_id: str,
    months: int = 12,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    if udf.empty:
        return ""

    daily = udf.set_index("transaction_date").resample("D")["transaction_amount"].sum().fillna(0)
    cum_exp = daily.where(daily > 0, 0).cumsum()
    cum_inc = daily.where(daily < 0, 0).abs().cumsum()
    user_name = udf["user_name"].iloc[0]

    fig = go.Figure()

    # Income area
    fig.add_trace(go.Scatter(
        x=cum_inc.index, y=cum_inc.values,
        mode="lines", name="Cumulative Income",
        line=dict(color=_GREEN, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,214,143,0.10)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Income: $%{y:,.0f}<extra></extra>",
    ))

    # Expense area
    fig.add_trace(go.Scatter(
        x=cum_exp.index, y=cum_exp.values,
        mode="lines", name="Cumulative Expenses",
        line=dict(color=_RED, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.10)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Expenses: $%{y:,.0f}<extra></extra>",
    ))

    # Net savings line
    net = cum_inc.values - cum_exp.values
    fig.add_trace(go.Scatter(
        x=cum_inc.index, y=net,
        mode="lines", name="Net Position",
        line=dict(color=_ORANGE, width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Net: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        f"Cumulative Income vs Expenses — {user_name}",
        xaxis_title="Date",
        yaxis_title="Cumulative Amount ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        xaxis_tickformat="%b %Y",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
    ))

    return _save(fig, user_name, "cumulative_flow")

# ═══════════════════════════════════════════════════════════════════
#  9.  Monthly Heatmap (Category × Month grid)
# ═══════════════════════════════════════════════════════════════════

def plot_monthly_heatmap(
    df: pd.DataFrame,
    user_id: str,
    months: int = 6,
    top_n: int = 8,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    pivot = (
        expenses.set_index("transaction_date")
        .groupby([pd.Grouper(freq="ME"), "transaction_category_detail"])["transaction_amount"]
        .sum()
        .unstack(fill_value=0)
    )
    if pivot.empty:
        return ""

    cat_totals = pivot.sum().nlargest(top_n)
    plot_df = pivot[cat_totals.index].copy()
    plot_df.columns = [friendly_category(c) for c in plot_df.columns]
    month_labels = [d.strftime("%b %Y") for d in plot_df.index]
    user_name = udf["user_name"].iloc[0]

    fig = go.Figure(go.Heatmap(
        z=plot_df.values.T,
        x=month_labels,
        y=list(plot_df.columns),
        colorscale=[[0, _BG], [0.3, "#1a1a4e"], [0.6, "#6C5CE7"], [1, "#A78BFA"]],
        hovertemplate="<b>%{y}</b><br>%{x}: $%{z:,.0f}<extra></extra>",
        texttemplate="$%{z:,.0f}",
        textfont=dict(size=10, color=_TEXT),
    ))

    fig.update_layout(**_base_layout(
        f"Spending Heatmap — {user_name}",
        xaxis_title="Month",
        height=max(450, top_n * 50),
        margin=dict(l=140, r=30, t=60, b=50),
    ))

    return _save(fig, user_name, "monthly_heatmap")

# ═══════════════════════════════════════════════════════════════════
#  10.  Savings Ratio (Monthly savings rate line)
# ═══════════════════════════════════════════════════════════════════

def plot_savings_ratio(
    df: pd.DataFrame,
    user_id: str,
    months: int = 8,
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

    # Savings rate = (income - expenses) / income * 100
    ratio = ((monthly_inc - monthly_exp) / monthly_inc.replace(0, np.nan) * 100).fillna(0)
    user_name = udf["user_name"].iloc[0]
    month_labels = [d.strftime("%b %Y") for d in idx]

    fig = go.Figure()

    # Bar for each month coloured by positive/negative
    colors = [_GREEN if r >= 0 else _RED for r in ratio.values]
    fig.add_trace(go.Bar(
        x=month_labels, y=ratio.values,
        marker=dict(color=colors, cornerradius=4),
        hovertemplate="<b>%{x}</b><br>Savings rate: %{y:.1f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in ratio.values],
        textposition="outside",
        textfont=dict(size=11, color=_TEXT_SEC),
    ))

    # 20% target line
    fig.add_hline(y=20, line=dict(color=_ORANGE, width=1.5, dash="dash"),
                  annotation_text="20% target",
                  annotation_font=dict(color=_ORANGE, size=10),
                  annotation_position="top right")
    fig.add_hline(y=0, line=dict(color=_GRID, width=1))

    fig.update_layout(**_base_layout(
        f"Monthly Savings Rate — {user_name}",
        xaxis_title="Month",
        yaxis_title="Savings Rate (%)",
        yaxis_ticksuffix="%",
    ))

    return _save(fig, user_name, "savings_ratio")

# ═══════════════════════════════════════════════════════════════════
#  11.  Transaction Scatter (amount vs date, coloured by category)
# ═══════════════════════════════════════════════════════════════════

def plot_transaction_scatter(
    df: pd.DataFrame,
    user_id: str,
    months: int = 6,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0].copy()
    if expenses.empty:
        return ""

    # Get top 5 categories for colour, rest = "Other"
    top_cats = expenses.groupby("transaction_category_detail")["transaction_amount"].sum().nlargest(5).index.tolist()
    expenses["cat_display"] = expenses["transaction_category_detail"].apply(
        lambda c: friendly_category(c) if c in top_cats else "Other"
    )
    user_name = udf["user_name"].iloc[0]

    fig = go.Figure()
    cats = [friendly_category(c) for c in top_cats] + ["Other"]
    for i, cat in enumerate(cats):
        subset = expenses[expenses["cat_display"] == cat]
        if subset.empty:
            continue
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=subset["transaction_date"],
            y=subset["transaction_amount"],
            mode="markers",
            name=cat,
            marker=dict(size=8, color=color, opacity=0.75, line=dict(width=1, color=_PAPER)),
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "%{x|%b %d, %Y}<br>"
                "$%{y:,.2f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(**_base_layout(
        f"Transaction Timeline — {user_name}",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        xaxis_tickformat="%b %d",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
    ))

    return _save(fig, user_name, "transaction_scatter")

# ═══════════════════════════════════════════════════════════════════
#  12.  Category Deep-Dive (top category merchant breakdown)
# ═══════════════════════════════════════════════════════════════════

def plot_category_deepdive(
    df: pd.DataFrame,
    user_id: str,
    category: Optional[str] = None,
    months: int = 12,
) -> str:
    udf = _user_df(df, user_id)
    udf = _filter_months(udf, months)
    expenses = udf[udf["transaction_amount"] > 0]
    if expenses.empty:
        return ""

    # Default to the top spending category
    if not category:
        category = expenses.groupby("transaction_category_detail")["transaction_amount"].sum().idxmax()

    cat_expenses = expenses[expenses["transaction_category_detail"] == category]
    if cat_expenses.empty:
        return ""

    user_name = udf["user_name"].iloc[0]
    cat_label = friendly_category(category)

    # Merchant breakdown within this category
    merch = cat_expenses.groupby("merchant_name")["transaction_amount"].agg(["sum", "count"]).nlargest(8, "sum")
    total_cat = cat_expenses["transaction_amount"].sum()

    merch_sorted = merch.iloc[::-1]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(merch_sorted))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=merch_sorted.index,
        x=merch_sorted["sum"].values,
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"${v:,.0f} ({int(c)} txns)" for v, c in zip(merch_sorted["sum"].values, merch_sorted["count"].values)],
        textposition="outside",
        textfont=dict(color=_TEXT_SEC, size=10),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))

    fig.add_annotation(
        text=f"Total: ${total_cat:,.0f}",
        xref="paper", yref="paper", x=0.98, y=1.05,
        showarrow=False,
        font=dict(size=13, color=_ACCENT),
    )

    fig.update_layout(**_base_layout(
        f"{cat_label} Deep Dive — {user_name}",
        xaxis_title="Amount ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        height=max(450, len(merch) * 50),
        margin=dict(l=120, r=100, t=70, b=50),
    ))

    return _save(fig, user_name, "category_deepdive")

# ═══════════════════════════════════════════════════════════════════
#  Dispatch map
# ═══════════════════════════════════════════════════════════════════

VIZ_FUNCTIONS = {
    "plot_monthly_spending_trend": plot_monthly_spending_trend,
    "plot_category_breakdown": plot_category_breakdown,
    "plot_income_vs_expense": plot_income_vs_expense,
    "plot_top_merchants": plot_top_merchants,
    "plot_weekly_pattern": plot_weekly_pattern,
    "plot_category_trends": plot_category_trends,
    "plot_expense_distribution": plot_expense_distribution,
    "plot_cumulative_flow": plot_cumulative_flow,
    "plot_monthly_heatmap": plot_monthly_heatmap,
    "plot_savings_ratio": plot_savings_ratio,
    "plot_transaction_scatter": plot_transaction_scatter,
    "plot_category_deepdive": plot_category_deepdive,
}
