"""Main TransactionRAGPipeline — the 4-stage orchestrator."""

import json
import os
import re
import time
import logging
from typing import Optional

import pandas as pd

from config import MAX_TOKENS_INPUT
from cache import KVCache
from guardrails import (
    run_input_guardrails,
    run_output_guardrails,
    AuditLogger,
    enforce_token_budget,
)
from visualizations import TOOL_SCHEMAS, VIZ_FUNCTIONS
from llm_client import LLMClient
from categories import CATEGORY_DISPLAY as _CATEGORY_DISPLAY, friendly_category

logger = logging.getLogger("pipeline")


class TransactionRAGPipeline:
    """Production-grade AI pipeline for financial transaction analysis.

    Usage
    -----
    pipeline = TransactionRAGPipeline(df=transactions_df)
    result = pipeline.run(user_id="usr_a1b2c3d4",
                          prompt="What did I spend the most on last month?")
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        self.df = df
        self._all_user_ids = set(df["user_id"].unique())

        self.cache = KVCache()
        self.llm = LLMClient()
        self.audit = AuditLogger()

    # ── helpers ──

    def _user_df(self, user_id: str) -> Optional[pd.DataFrame]:
        udf = self.df[self.df["user_id"] == user_id]
        return udf.copy() if not udf.empty else None

    def _category_mapping_str(self) -> str:
        """Build a dynamic category mapping string from actual data categories."""
        raw_cats = self.df["transaction_category_detail"].dropna().unique()
        pairs = [f"{cat} → {friendly_category(cat)}" for cat in sorted(raw_cats)]
        return ", ".join(pairs)

    def _build_system_prompt(self, profile: dict, query_history: list, viz_state: dict = None) -> str:
        # Build a human-readable top-categories string
        top_cats_raw = profile.get("top_expense_categories", {})
        if isinstance(top_cats_raw, dict):
            top_cats_display = ", ".join(
                f"{friendly_category(cat)} (${amt:,.0f})"
                for cat, amt in list(top_cats_raw.items())[:5]
            ) or "N/A"
        else:
            top_cats_display = ", ".join(
                f"{friendly_category(c['category'])} (${c['amount']:,.0f})"
                for c in top_cats_raw[:5]
            ) or "N/A"

        # Compute derived financial signals for the prompt
        avg_monthly_spend = profile.get("avg_monthly_spend", 0)
        avg_monthly_income = profile.get("avg_monthly_income", 0)
        savings_rate = (
            round((avg_monthly_income - avg_monthly_spend) / avg_monthly_income * 100, 1)
            if avg_monthly_income > 0 else 0
        )
        savings_signal = (
            f"saving {savings_rate:.1f}% of monthly income — above the 20% benchmark ✓"
            if savings_rate >= 20
            else f"saving {savings_rate:.1f}% of monthly income — below the recommended 20% target"
            if savings_rate > 0
            else f"spending exceeds income by {abs(savings_rate):.1f}% — actively overspending"
        )

        system = (
            "You are Finley, an elite personal finance analyst embedded inside a real-time transaction "
            "intelligence platform. You have direct, live access to the user's complete bank and card "
            "transaction history. You are not a generic chatbot — you are a sharp, data-driven financial "
            "advisor who speaks plainly, thinks in benchmarks, and always leads with the most useful insight.\n\n"

            "Your character: You combine the precision of a quant analyst, the clarity of a great teacher, "
            "and the candour of a trusted friend. You are direct, never preachy. You celebrate good financial "
            "habits and flag risks honestly — but without judgement. You always know the *so what* behind the numbers.\n\n"

            "<core_mission>\n"
            "Your primary job is NOT to recite numbers — it is to turn raw transaction data into "
            "actionable financial intelligence. Every response must pass this test: "
            "\"Does this help the user make a better financial decision or understand something they didn't before?\"\n"
            "- Lead EVERY response with the single most useful insight or direct answer (1-2 sentences max).\n"
            "- Then support it with specific data points from the user's actual transactions.\n"
            "- End with an actionable takeaway, a notable pattern, or a smart follow-up question — "
            "something that moves the conversation forward.\n"
            "- NEVER pad responses with filler phrases like 'Great question!' or 'As an AI...'.\n"
            "- NEVER fabricate, estimate, or hallucinate figures. Every number must come from the user's data.\n"
            "</core_mission>\n\n"

            "<response_style>\n"
            "VOICE: Speak like a sharp financial friend — knowledgeable, direct, warm, never robotic.\n"
            "FORMAT:\n"
            "  - Short answers (1 clear stat): 2-3 sentences of prose. No bullets.\n"
            "  - Medium answers (comparing 2-5 items): Prose intro + tight bullet list.\n"
            "  - Long answers (deep analysis, full reports): Prose sections with bold labels. No markdown headers (# or ##).\n"
            "  - NEVER start a response with a heading, bold label, or markdown header.\n"
            "  - NEVER use emojis unless the user uses them first.\n"
            "NUMBERS:\n"
            "  - Always format currency as $X,XXX.XX (e.g. $1,234.56).\n"
            "  - Always format percentages to one decimal place (e.g. 23.4%).\n"
            "  - When a number is significant, say WHY it matters (e.g. 'that's 43% of your total spend').\n"
            "CATEGORY NAMES:\n"
            "  - NEVER use raw database column names (e.g. RENT_HOUSING, FOOD_GROCERY). "
            "Always convert to friendly names. Here is the full mapping for this dataset:\n"
            f"    {self._category_mapping_str()}\n"
            "  - General rule: take the first segment, capitalize naturally, drop the trailing group tag.\n"
            "</response_style>\n\n"

            "<financial_intelligence>\n"
            "Apply these financial benchmarks and frameworks when relevant — but only when supported by the data:\n"
            "  - 50/30/20 Rule: ~50% needs (housing, food, transport), ~30% wants (dining, entertainment), "
            "~20% savings/debt. Use this to contextualize the user's category split.\n"
            "  - Savings target: 20% of gross income is the standard recommendation. Flag deviations.\n"
            "  - Emergency fund: 3-6 months of expenses is a healthy baseline. "
            "If you can infer savings are low, gently note this.\n"
            "  - Spending spikes: A month 20%+ above average warrants a callout.\n"
            "  - Merchant concentration: If one merchant takes >15% of spend in a category, mention it.\n"
            "  - Positive reinforcement: Acknowledge improving trends, falling discretionary spend, or "
            "savings rate improvements — these are meaningful wins worth calling out.\n"
            "PROACTIVE INSIGHTS: Even when not asked, if you notice an anomaly, opportunity, or risk "
            "in the data context, briefly flag it at the end of your response with a natural transition "
            "like 'One thing I noticed...' or 'Worth flagging:'.\n"
            "</financial_intelligence>\n\n"

            "<data_conventions>\n"
            "- Positive transaction_amount = EXPENSE (money going out)\n"
            "- Negative transaction_amount = INCOME (money coming in: salary, refunds, cashback, transfers)\n"
            "- Available data columns: user_id, user_name, transaction_date, "
            "transaction_amount, transaction_category_detail, merchant_name\n"
            "- All dates are in the user's local timezone. Refer to months by name (e.g. 'March') not number.\n"
            "</data_conventions>\n\n"

            "<visualization_tools>\n"
            "You have twelve charting tools. Use them proactively — a good chart often explains in 2 seconds "
            "what takes 5 sentences to write. Always call charts when the question involves trends, comparisons, "
            "distributions, or any 'show me' phrasing. For comprehensive questions, call 2-3 complementary charts.\n"
            "  - plot_monthly_spending_trend(months, category_filter) → line chart of spending over time. "
            "Use for: trend questions, month-by-month changes, 'how has my spending changed?'\n"
            "  - plot_category_breakdown(period, top_n) → donut chart of category split. "
            "Use for: 'where does my money go?', category questions, budget reviews.\n"
            "  - plot_income_vs_expense(months, show_net_line) → grouped bar chart of income vs spend. "
            "Use for: savings analysis, net position, income questions.\n"
            "  - plot_top_merchants(top_n, months) → horizontal bar chart ranked by merchant spend. "
            "Use for: merchant questions, 'where do I shop?', retailer analysis.\n"
            "  - plot_weekly_pattern(months) → bar chart of avg daily spend by weekday. "
            "Use for: habit analysis, 'when do I spend most?', weekly pattern questions.\n"
            "  - plot_category_trends(months, top_n) → stacked area chart of categories over time. "
            "Use for: how category mix is shifting, long-term trend analysis.\n"
            "  - plot_expense_distribution(bins) → histogram of transaction sizes. "
            "Use for: 'how big are my purchases?', outlier detection, spending behavior.\n"
            "  - plot_cumulative_flow(months) → cumulative income vs expense area chart. "
            "Use for: net wealth trajectory, 'am I getting ahead or behind?'\n"
            "  - plot_monthly_heatmap(months, top_n) → category × month spending heatmap. "
            "Use for: pattern recognition, 'which months are expensive in which categories?'\n"
            "  - plot_savings_ratio(months) → monthly savings rate vs 20% target. "
            "Use for: savings rate analysis, 'how well am I saving?', financial health.\n"
            "  - plot_transaction_scatter(months) → all transactions plotted by date and amount. "
            "Use for: outlier detection, 'what were my biggest purchases?', transaction overview.\n"
            "  - plot_category_deepdive(category, months) → merchant breakdown within one category. "
            "Use for: 'what am I spending on food?', single-category deep dives.\n"
            "Adjust the 'months' parameter to match the user's time reference: "
            "'last month' → 1, 'last quarter' → 3, 'last 6 months' → 6, 'last year' → 12, 'all time' → 24.\n"
            "</visualization_tools>\n\n"

            "<follow_up_handling>\n"
            "When the user asks to EXPLAIN in words ('tell me more', 'elaborate', 'what does that mean', "
            "'break it down', 'explain this', 'in plain English', 'analyze this', 'why is that'):\n"
            "  - Do NOT call any chart tools. Zero charts.\n"
            "  - Write a rich narrative that unpacks the data: what the numbers mean, what's driving them, "
            "what's surprising, what it implies for their financial health, and what they could do about it.\n"
            "  - Compare months to each other. Flag meaningful patterns. Name specific categories and merchants.\n"
            "  - Use the 50/30/20 benchmark or savings rate context where helpful.\n\n"
            "When the user asks for a BETTER or DIFFERENT chart ('improve this', 'more detail', "
            "'different chart', 'redo this', 'show me more'):\n"
            "  - Reference the last chart generated (see <last_visualization> below).\n"
            "  - Generate it with meaningfully different parameters: wider time range, more categories, "
            "a different chart type, or a category filter. Explain the change in one sentence.\n"
            "  - Do NOT repeat the same chart with the same parameters.\n"
            "</follow_up_handling>\n\n"

            f"<user_financial_profile>\n"
            f"Name: {profile['user_name']}\n"
            f"Analysis period: {profile['date_range']['start']} → {profile['date_range']['end']}\n"
            f"Total transactions analyzed: {profile['total_transactions']:,}\n"
            f"Average monthly spending: ${avg_monthly_spend:,.2f}\n"
            f"Average monthly income: ${avg_monthly_income:,.2f}\n"
            f"Financial position: Currently {savings_signal}\n"
            f"Top expense categories: {top_cats_display}\n"
            f"</user_financial_profile>\n"
        )

        if viz_state:
            system += (
                f"\n<last_visualization>\n"
                f"Chart type: {viz_state.get('last_chart_type', 'unknown')}\n"
                f"Parameters used: {json.dumps(viz_state.get('filters', {}))}\n"
                "If the user asks to improve, redo, or show a different chart, "
                "generate it with meaningfully different parameters (wider date range, more categories, different chart type, "
                "or a category filter). Always explain what you changed and why in one sentence.\n"
                "</last_visualization>\n"
            )

        if query_history:
            system += "\n<conversation_context>\n"
            system += "Recent questions from this session (use these to understand follow-up intent):\n"
            for i, q in enumerate(query_history[-5:], 1):
                system += f"{i}. Q: \"{q['prompt']}\" → Summary: {q['result_summary']}\n"
            system += "</conversation_context>\n"

        return system

    def _data_summary(self, user_df: pd.DataFrame) -> dict:
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]

        monthly_exp = (
            expenses.set_index("transaction_date")
            .resample("ME")["transaction_amount"]
            .sum()
        )

        return {
            "total_expenses": round(expenses["transaction_amount"].sum(), 2),
            "total_income": round(abs(income["transaction_amount"].sum()), 2),
            "num_transactions": len(user_df),
            "monthly_expenses": {
                str(k.date()): round(v, 2) for k, v in monthly_exp.items()
            },
            "category_totals": {
                friendly_category(k): v
                for k, v in expenses.groupby("transaction_category_detail")["transaction_amount"]
                .sum()
                .to_dict()
                .items()
            },
        }

    def _execute_tool_calls(self, tool_calls: list, user_id: str) -> list[str]:
        paths: list[str] = []
        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            if fn_name not in VIZ_FUNCTIONS:
                logger.warning(f"Unknown tool: {fn_name}")
                continue

            try:
                raw_args = tc.get("function", {}).get("arguments", "{}")
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON args for {fn_name}")
                continue

            args["user_id"] = user_id  # enforce — never trust LLM's user_id

            try:
                path = VIZ_FUNCTIONS[fn_name](df=self.df, **args)
                if path:
                    paths.append(path)
                    self.cache.set_viz_state(
                        user_id,
                        chart_type=fn_name,
                        axes=args,
                        filters={k: v for k, v in args.items() if k != "user_id"},
                    )
            except Exception as exc:
                logger.error(f"Tool exec error ({fn_name}): {exc}")

        return paths

    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip spurious <tool_call> XML and convert raw category names to friendly labels."""
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL)  # unclosed
        # Replace raw DB category names (e.g. RENT_HOUSING) with friendly labels
        for raw, friendly in _CATEGORY_DISPLAY.items():
            text = text.replace(raw, friendly)
        return text.strip()

    def _summarise_from_data(self, user_df: pd.DataFrame, prompt: str, chart_paths: list[str]) -> str:
        """Generate a narrative, insight-driven textual answer from the DataFrame when the LLM
        produced only tool calls (no usable text). Reads like a financial analyst, not a data dump."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]
        prompt_lower = prompt.lower()

        total_exp = round(expenses["transaction_amount"].sum(), 2)
        total_inc = round(abs(income["transaction_amount"].sum()), 2)
        net = total_inc - total_exp
        savings_rate = (net / total_inc * 100) if total_inc > 0 else 0

        lines: list[str] = []

        if any(w in prompt_lower for w in ["most", "top", "highest", "breakdown", "category", "where"]):
            top = (
                expenses.groupby("transaction_category_detail")["transaction_amount"]
                .sum().nlargest(5)
            )
            if not top.empty:
                top_name = friendly_category(top.index[0])
                top_pct = (top.iloc[0] / total_exp * 100) if total_exp else 0
                lines.append(
                    f"Your biggest spending category is **{top_name}** at ${top.iloc[0]:,.2f} "
                    f"— that's {top_pct:.1f}% of everything you've spent. Here's the full top-5 breakdown:"
                )
                for cat, amt in top.items():
                    pct = (amt / total_exp * 100) if total_exp else 0
                    lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
                top3_pct = sum((v / total_exp * 100) for v in top.iloc[:3].values) if total_exp else 0
                lines.append(
                    f"\nYour top 3 categories together account for {top3_pct:.0f}% of all spending — "
                    f"a useful starting point if you're thinking about where to cut back."
                )

        elif any(w in prompt_lower for w in ["trend", "over time", "month"]):
            monthly = (
                expenses.set_index("transaction_date")
                .resample("ME")["transaction_amount"].sum()
            )
            if not monthly.empty and len(monthly) >= 2:
                avg = monthly.mean()
                high_m = monthly.idxmax().strftime("%b %Y")
                low_m = monthly.idxmin().strftime("%b %Y")
                change = monthly.iloc[-1] - monthly.iloc[-2]
                direction = "up" if change > 0 else "down"
                lines.append(
                    f"Your monthly spending averages ${avg:,.2f}, with a peak in {high_m} "
                    f"(${monthly.max():,.2f}) and a low in {low_m} (${monthly.min():,.2f}). "
                    f"Last month's spending moved {direction} by ${abs(change):,.2f}."
                )
                lines.append("\nMonth-by-month breakdown:")
                for dt, amt in monthly.items():
                    flag = " ← above average" if amt > avg * 1.2 else ""
                    lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f}{flag}")
            elif not monthly.empty:
                lines.append(f"Only one month of data so far — spending was ${monthly.iloc[0]:,.2f}.")

        elif any(w in prompt_lower for w in ["saving", "save", "income", "expense", "financial", "net"]):
            lines.append(
                f"Across all your transactions, you've earned ${total_inc:,.2f} and spent ${total_exp:,.2f} — "
                f"leaving you {'ahead' if net >= 0 else 'behind'} by ${abs(net):,.2f}."
            )
            if savings_rate >= 20:
                lines.append(
                    f"Your overall savings rate of {savings_rate:.1f}% is above the recommended 20% — solid financial discipline."
                )
            elif savings_rate > 0:
                lines.append(
                    f"Your overall savings rate is {savings_rate:.1f}%. The standard target is 20% — there's room to grow, "
                    f"and even a small reduction in discretionary spending can move that needle."
                )
            else:
                lines.append(
                    f"You're currently spending more than you earn. Identifying and trimming the top discretionary "
                    f"categories would be the fastest path to getting back on track."
                )

        else:
            lines.append(
                f"Here's the quick picture: {len(user_df):,} total transactions, "
                f"${total_exp:,.2f} spent and ${total_inc:,.2f} earned — a net of "
                f"{'savings' if net >= 0 else 'deficit'} of ${abs(net):,.2f}."
            )

        if chart_paths:
            n = len(chart_paths)
            lines.append(f"\nI've generated {n} chart{'s' if n > 1 else ''} below to visualize this for you.")
        return "\n".join(lines)

    def _fallback_response(self, user_df: pd.DataFrame, prompt: str, user_id: str,
                            conversation: list = None) -> tuple[str, list[str]]:
        """Generate a data-driven, question-aware answer + charts when the LLM is unreachable.
        Handles diverse questions including follow-ups, merchant queries, specific categories,
        graph requests, and more. Returns (response_text, chart_paths)."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]
        user_name = user_df["user_name"].iloc[0]
        prompt_lower = prompt.lower()
        lines: list[str] = []
        chart_paths: list[str] = []

        total_exp = round(expenses["transaction_amount"].sum(), 2)
        total_inc = round(abs(income["transaction_amount"].sum()), 2)
        net_savings = round(total_inc - total_exp, 2)

        monthly_exp = (
            expenses.set_index("transaction_date")
            .resample("ME")["transaction_amount"].sum()
        )
        monthly_inc = (
            income.set_index("transaction_date")
            .resample("ME")["transaction_amount"].sum().abs()
        )
        last_month_name = monthly_exp.index[-1].strftime("%b %Y") if not monthly_exp.empty else "N/A"
        last_month_amt = round(monthly_exp.iloc[-1], 2) if not monthly_exp.empty else 0
        num_months = max(len(monthly_exp), 1)

        top_cats = (
            expenses.groupby("transaction_category_detail")["transaction_amount"]
            .sum().nlargest(5)
        )
        all_cats = expenses.groupby("transaction_category_detail")["transaction_amount"].sum().sort_values(ascending=False)

        # Get viz state for follow-up context
        viz_state = self.cache.get_viz_state(user_id)
        last_chart = viz_state.get("last_chart_type", "") if viz_state else ""

        # ── Helper to safely generate charts ──
        def _try_chart(fn_name: str, **kwargs):
            try:
                p = VIZ_FUNCTIONS[fn_name](df=self.df, user_id=user_id, **kwargs)
                if p: chart_paths.append({"path": p, "fn": fn_name, "kwargs": kwargs})
            except Exception:
                pass

        # ── Detect specific merchant queries ──
        merchants = user_df["merchant_name"].str.lower().unique().tolist()
        matched_merchant = None
        for m in merchants:
            if m and m in prompt_lower:
                matched_merchant = m
                break

        # ── Detect specific category queries ──
        categories = user_df["transaction_category_detail"].str.lower().unique().tolist()
        matched_category = None
        # Common financial terms that appear as category parts but shouldn't trigger
        # category matching when used in general questions
        _cat_stopwords = {
            "income", "expense", "expenses", "save", "saving", "savings",
            "compare", "total", "amount", "budget", "money", "trend",
            "monthly", "daily", "weekly", "average", "payment",
        }
        for cat in categories:
            # Check both full name and parts
            cat_parts = cat.split("_")
            if cat in prompt_lower:
                matched_category = cat
                break
            for part in cat_parts:
                if len(part) > 3 and part in prompt_lower and part not in _cat_stopwords:
                    matched_category = cat
                    break
            if matched_category:
                break

        # ══════════════════════════════════════════════════════════
        #  INTENT DETECTION (ordered: specific → general)
        # ══════════════════════════════════════════════════════════

        if any(w in prompt_lower for w in ["explain", "in words", "elaborate", "tell me more",
                                            "what does that mean", "what does this mean",
                                            "summarize", "summarise", "interpret", "plain english",
                                            "describe", "narrative", "insight", "why is",
                                            "break it down", "walk me through", "analyze this"]):
            # TEXT-ONLY follow-up: explain the previous data in words, NO charts
            last_assistant = ""
            second_last_assistant = ""
            if conversation:
                assistant_msgs = [m.get("content", "") for m in conversation if m.get("role") == "assistant"]
                if assistant_msgs:
                    last_assistant = assistant_msgs[-1]
                if len(assistant_msgs) >= 2:
                    second_last_assistant = assistant_msgs[-2]

            # Detect if the last response was ALREADY an explanation (narrative text).
            # If so, "tell me more" should provide DEEPER detail, not repeat.
            already_explained = any(phrase in last_assistant.lower() for phrase in [
                "looking at your spending over",
                "here's what your category breakdown",
                "here's the picture on your savings",
                "here's what your merchant spending",
                "in a nutshell",
            ])

            # When already explained, look at the second-to-last message for the original topic
            context_msg = second_last_assistant if already_explained else last_assistant
            context_lower = context_msg.lower()

            if already_explained and ("trend" in context_lower or "monthly" in context_lower):
                # DEEPER trend analysis: month-over-month changes, category breakdown per month
                lines.append("Digging deeper into your spending trends:\n")
                if not monthly_exp.empty and len(monthly_exp) >= 2:
                    lines.append("**Month-over-month changes:**")
                    prev = None
                    for dt, amt in monthly_exp.items():
                        if prev is not None:
                            diff = amt - prev
                            pct_change = (diff / prev * 100) if prev else 0
                            arrow = "↑" if diff > 0 else "↓"
                            lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f} ({arrow} ${abs(diff):,.0f}, {abs(pct_change):.1f}%)")
                        else:
                            lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f} (baseline)")
                        prev = amt
                    # Category shifts
                    lines.append(f"\n**What's driving the changes:**")
                    cat_by_month = expenses.set_index("transaction_date").groupby(
                        [pd.Grouper(freq="ME"), "transaction_category_detail"]
                    )["transaction_amount"].sum().unstack(fill_value=0)
                    if len(cat_by_month) >= 2:
                        last_m = cat_by_month.iloc[-1]
                        prev_m = cat_by_month.iloc[-2]
                        diffs = (last_m - prev_m).sort_values()
                        biggest_increase = diffs.iloc[-1]
                        biggest_decrease = diffs.iloc[0]
                        if biggest_increase > 0:
                            lines.append(f"  • Biggest increase: **{friendly_category(diffs.index[-1])}** "
                                         f"(+${biggest_increase:,.0f})")
                        if biggest_decrease < 0:
                            lines.append(f"  • Biggest decrease: **{friendly_category(diffs.index[0])}** "
                                         f"(-${abs(biggest_decrease):,.0f})")
                    # Spending velocity
                    if len(monthly_exp) >= 3:
                        recent_avg = monthly_exp.iloc[-3:].mean()
                        earlier_avg = monthly_exp.iloc[:-3].mean() if len(monthly_exp) > 3 else monthly_exp.iloc[0]
                        if recent_avg > earlier_avg:
                            lines.append(f"\nYour recent 3-month average (${recent_avg:,.0f}) is higher than your "
                                         f"earlier average (${earlier_avg:,.0f}) — spending is trending upward.")
                        else:
                            lines.append(f"\nYour recent 3-month average (${recent_avg:,.0f}) is lower than your "
                                         f"earlier average (${earlier_avg:,.0f}) — spending is trending downward. Nice!")
                else:
                    lines.append("Not enough monthly data for a detailed deep-dive.")

            elif already_explained:
                # Already explained something else — provide a different angle
                lines.append(f"Here's another angle on your finances:\n")
                # Show what hasn't been covered yet
                if "merchant" not in last_assistant.lower() and "merchant" not in context_lower:
                    top_m = expenses.groupby("merchant_name")["transaction_amount"].agg(["sum", "count"]).nlargest(5, "sum")
                    if not top_m.empty:
                        lines.append("**Your top merchants:**")
                        for name, row in top_m.iterrows():
                            lines.append(f"  • {name}: ${row['sum']:,.2f} ({int(row['count'])} visits)")
                if "saving" not in last_assistant.lower() and "saving" not in context_lower:
                    savings_rate = (net_savings / total_inc * 100) if total_inc else 0
                    lines.append(f"\n**Quick savings check:** You've saved ${net_savings:,.2f} "
                                 f"({savings_rate:.1f}% of income).")
                if not lines:
                    # If everything's been covered, give actionable tips
                    lines.append("Based on everything we've looked at, here are some quick takeaways:")
                    lines.append(f"  • Your #1 expense is {friendly_category(top_cats.index[0])} at ${top_cats.iloc[0]:,.2f}")
                    avg_txn = expenses["transaction_amount"].mean()
                    lines.append(f"  • Your average transaction is ${avg_txn:,.2f}")
                    lines.append(f"  • You make about {round(len(expenses)/num_months)} transactions per month")

            elif "trend" in context_lower or "monthly" in context_lower:
                # Explain spending trend narratively
                if not monthly_exp.empty and len(monthly_exp) >= 2:
                    avg = monthly_exp.mean()
                    highest_month = monthly_exp.idxmax().strftime("%b %Y")
                    lowest_month = monthly_exp.idxmin().strftime("%b %Y")
                    highest_val = monthly_exp.max()
                    lowest_val = monthly_exp.min()
                    last_val = monthly_exp.iloc[-1]
                    prev_val = monthly_exp.iloc[-2]
                    change = last_val - prev_val
                    direction = "went up" if change > 0 else "went down"

                    lines.append(f"Looking at your spending over the past {len(monthly_exp)} months, "
                                 f"your average monthly spend is **${avg:,.0f}**.")
                    lines.append(f"")
                    lines.append(f"Your **highest spending month** was **{highest_month}** at **${highest_val:,.2f}**, "
                                 f"while your **lowest** was **{lowest_month}** at **${lowest_val:,.2f}** — "
                                 f"a difference of ${highest_val - lowest_val:,.2f}.")
                    lines.append(f"")
                    lines.append(f"Most recently, your spending {direction} by **${abs(change):,.2f}** "
                                 f"from {monthly_exp.index[-2].strftime('%b')} to {monthly_exp.index[-1].strftime('%b')}.")

                    # Flag months that are above average
                    above_avg = [(dt.strftime("%b %Y"), amt) for dt, amt in monthly_exp.items() if amt > avg * 1.2]
                    if above_avg:
                        spikes = ", ".join(f"{m} (${a:,.0f})" for m, a in above_avg)
                        lines.append(f"")
                        lines.append(f"Months with notably higher spending (20%+ above your average): {spikes}. "
                                     f"It might be worth checking what drove those spikes — large one-off purchases, "
                                     f"travel, or seasonal expenses.")

                    # Top category context
                    if not top_cats.empty:
                        lines.append(f"")
                        top_cat_name = friendly_category(top_cats.index[0])
                        top_cat_pct = (top_cats.iloc[0] / total_exp * 100) if total_exp else 0
                        lines.append(f"Your biggest spending category overall is **{top_cat_name}** "
                                     f"({top_cat_pct:.0f}% of total), so fluctuations there likely "
                                     f"drive most of the month-to-month variation.")
                else:
                    lines.append("There isn't enough monthly data yet to provide a detailed trend explanation.")

            elif "category" in context_lower or "breakdown" in context_lower:
                # Explain category breakdown narratively
                if not all_cats.empty:
                    top3 = all_cats.head(3)
                    bottom3 = all_cats.tail(3)
                    lines.append(f"Here's what your category breakdown tells us:")
                    lines.append(f"")
                    top3_pct = sum((v / total_exp * 100) for v in top3.values)
                    top3_names = ", ".join(friendly_category(c) for c in top3.index)
                    lines.append(f"Your top 3 categories — **{top3_names}** — account for "
                                 f"**{top3_pct:.0f}%** of all spending. That's where the bulk of your money goes.")
                    lines.append(f"")
                    bottom_names = ", ".join(f"{friendly_category(c)} (${v:,.0f})" for c, v in bottom3.items())
                    lines.append(f"On the other end, your smallest categories are {bottom_names} — "
                                 f"these are relatively minor in your overall budget.")
                    if top3.iloc[0] / total_exp > 0.5:
                        lines.append(f"")
                        lines.append(f"**{friendly_category(top3.index[0])}** alone makes up over half your spending. "
                                     f"If you're looking to cut costs, that's the biggest lever — though it may be "
                                     f"a fixed expense like rent that's harder to reduce.")

            elif "saving" in context_lower or "income" in context_lower:
                # Explain savings narratively
                savings_rate = (net_savings / total_inc * 100) if total_inc else 0
                monthly_savings = net_savings / num_months if num_months else 0
                lines.append(f"Here's the picture on your savings:")
                lines.append(f"")
                if net_savings > 0:
                    lines.append(f"You're in positive territory — you've saved **${net_savings:,.2f}** overall, "
                                 f"which works out to about **${monthly_savings:,.0f}/month**. "
                                 f"Your savings rate of **{savings_rate:.1f}%** is "
                                 f"{'solid' if savings_rate >= 20 else 'a reasonable start'}.")
                    if savings_rate < 20:
                        lines.append(f"")
                        lines.append(f"Financial advisors often recommend saving at least 20% of income. "
                                     f"You're at {savings_rate:.1f}%, so there's room to grow — "
                                     f"even small cutbacks in discretionary categories could help.")
                else:
                    lines.append(f"You're currently spending **${abs(net_savings):,.2f}** more than you earn. "
                                 f"That's about **${abs(monthly_savings):,.0f}/month** in the red.")
                    lines.append(f"")
                    lines.append(f"Look at your top discretionary categories to find quick wins for cutting back.")

            elif "merchant" in context_lower:
                # Explain merchant data narratively
                top_m = expenses.groupby("merchant_name")["transaction_amount"].agg(["sum", "count"]).nlargest(5, "sum")
                if not top_m.empty:
                    lines.append(f"Here's what your merchant spending says about your habits:")
                    lines.append(f"")
                    top_name = top_m.index[0]
                    top_total = top_m.iloc[0]["sum"]
                    top_count = int(top_m.iloc[0]["count"])
                    lines.append(f"**{top_name}** is your #1 merchant at **${top_total:,.2f}** across {top_count} transactions. "
                                 f"That's ${top_total/top_count:,.0f} per visit on average.")
                    if len(top_m) >= 3:
                        others = ", ".join(f"{n} (${r['sum']:,.0f})" for n, r in top_m.iloc[1:4].iterrows())
                        lines.append(f"")
                        lines.append(f"Other frequent spots: {others}.")

            else:
                # Generic narrative explanation
                lines.append(f"In a nutshell: you've earned **${total_inc:,.2f}** and spent **${total_exp:,.2f}** "
                             f"across {len(user_df)} transactions.")
                lines.append(f"")
                if net_savings > 0:
                    lines.append(f"That leaves you **${net_savings:,.2f} ahead** — you're spending less than you earn, which is great.")
                else:
                    lines.append(f"You're currently **${abs(net_savings):,.2f} over budget** — spending more than income.")
                if not top_cats.empty:
                    lines.append(f"")
                    lines.append(f"Most of your money goes to **{friendly_category(top_cats.index[0])}** — "
                                 f"that single category is {top_cats.iloc[0]/total_exp*100:.0f}% of all spending.")

            # NO charts for text explanation follow-ups

        elif any(w in prompt_lower for w in ["better graph", "better chart", "improve", "redo", "regenerate",
                                            "different chart", "another graph", "new graph", "new chart",
                                            "clearer", "detailed graph", "detailed chart"]):
            # Follow-up: improve/redo last chart
            if last_chart == "plot_category_breakdown":
                lines.append("Here's an expanded category breakdown showing your full top 10, paired with the trend to show how the mix is shifting:")
                _try_chart("plot_category_breakdown", top_n=10, period="all_time")
                _try_chart("plot_monthly_spending_trend")
            elif last_chart == "plot_monthly_spending_trend":
                lines.append("Here's the expanded 12-month trend — more history gives you a clearer picture of seasonal patterns and trajectory:")
                _try_chart("plot_monthly_spending_trend", months=12)
                _try_chart("plot_category_breakdown")
            elif last_chart == "plot_income_vs_expense":
                lines.append("Here's a fuller income vs expense view covering the past year, plus your cumulative cash flow to show the overall trajectory:")
                _try_chart("plot_income_vs_expense", months=12)
                _try_chart("plot_monthly_spending_trend")
            else:
                lines.append("Here's a comprehensive view of your finances from multiple angles:")
                _try_chart("plot_category_breakdown", top_n=10)
                _try_chart("plot_monthly_spending_trend", months=12)
                _try_chart("plot_income_vs_expense")

        elif matched_merchant:
            # Merchant-specific query
            m_txns = user_df[user_df["merchant_name"].str.lower() == matched_merchant]
            m_exp = m_txns[m_txns["transaction_amount"] > 0]
            m_name = m_txns["merchant_name"].iloc[0] if not m_txns.empty else matched_merchant
            if not m_exp.empty:
                m_total = m_exp["transaction_amount"].sum()
                m_avg = m_exp["transaction_amount"].mean()
                m_pct = (m_total / total_exp * 100) if total_exp else 0
                m_cat = friendly_category(m_exp["transaction_category_detail"].mode().iloc[0])
                lines.append(
                    f"You've spent **${m_total:,.2f}** at **{m_name}** across {len(m_exp)} transactions "
                    f"— that's {m_pct:.1f}% of your total spend, averaging ${m_avg:,.2f} per visit. "
                    f"It falls under your **{m_cat}** category."
                )
            else:
                lines.append(f"Found {len(m_txns)} transactions linked to {m_name}:")
            if not m_txns.empty:
                date_range = f"{m_txns['transaction_date'].min().strftime('%b %d, %Y')} – {m_txns['transaction_date'].max().strftime('%b %d, %Y')}"
                lines.append(f"\nTransaction history at {m_name} ({date_range}):")
                for _, row in m_txns.nlargest(5, "transaction_date").iterrows():
                    sign = "+" if row["transaction_amount"] < 0 else "-"
                    lines.append(f"  • {row['transaction_date'].strftime('%b %d')}: {sign}${abs(row['transaction_amount']):,.2f}")
            _try_chart("plot_top_merchants")
            _try_chart("plot_monthly_spending_trend", category_filter=m_exp["transaction_category_detail"].mode().iloc[0] if not m_exp.empty else None)

        elif matched_category:
            # Category-specific query
            cat_upper = matched_category.upper()
            cat_txns = expenses[expenses["transaction_category_detail"].str.upper() == cat_upper]
            cat_label = friendly_category(cat_upper)
            if not cat_txns.empty:
                cat_total = cat_txns["transaction_amount"].sum()
                cat_avg = cat_txns["transaction_amount"].mean()
                pct = (cat_total / total_exp * 100) if total_exp else 0
                lines.append(
                    f"You've spent **${cat_total:,.2f}** on **{cat_label}** across {len(cat_txns)} transactions "
                    f"— that's {pct:.1f}% of your total spending, averaging ${cat_avg:,.2f} per transaction."
                )
                cat_monthly = cat_txns.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
                if not cat_monthly.empty and len(cat_monthly) >= 2:
                    cat_avg_monthly = cat_monthly.mean()
                    cat_high = cat_monthly.idxmax().strftime("%b %Y")
                    lines.append(
                        f"Your peak month for {cat_label} was {cat_high} (${cat_monthly.max():,.2f}), "
                        f"against a monthly average of ${cat_avg_monthly:,.2f}."
                    )
                    lines.append("\nMonthly breakdown:")
                    for dt, amt in cat_monthly.items():
                        flag = " ← high" if amt > cat_avg_monthly * 1.2 else ""
                        lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f}{flag}")
                if "merchant_name" in cat_txns.columns:
                    top_merch = cat_txns.groupby("merchant_name")["transaction_amount"].sum().nlargest(3)
                    if not top_merch.empty:
                        lines.append(f"\nTop merchants in {cat_label}:")
                        for merch, amt in top_merch.items():
                            m_pct = (amt / cat_total * 100) if cat_total else 0
                            lines.append(f"  • {merch}: ${amt:,.2f} ({m_pct:.1f}% of {cat_label} spend)")
                _try_chart("plot_monthly_spending_trend", category_filter=cat_upper.split("_")[-1])
            else:
                lines.append(f"No transactions found in the {cat_label} category.")

        elif any(w in prompt_lower for w in ["how should", "advice", "suggest", "recommend", "plan", "budget", "next month",
                                              "what should", "can i afford", "afford", "reduce", "cut"]):
            avg_monthly = round(monthly_exp.mean(), 2) if not monthly_exp.empty else 0
            avg_inc = round(total_inc / num_months, 2)
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            target_savings = round(avg_inc * 0.20, 2)
            current_monthly_savings = round(net_savings / num_months, 2) if num_months else 0
            lines.append(
                f"Based on your actual spending history, here's a realistic budget picture for {user_name}. "
                f"You bring in ~${avg_inc:,.2f}/month and spend ~${avg_monthly:,.2f}/month — "
                f"{'leaving ${:,.2f}/month as savings.'.format(current_monthly_savings) if current_monthly_savings >= 0 else 'a monthly shortfall of ${:,.2f}.'.format(abs(current_monthly_savings))}"
            )
            lines.append("")
            lines.append("Your current monthly spend by category (these are your actual numbers, not estimates):")
            for cat, amt in top_cats.items():
                avg_cat = round(amt / num_months, 2)
                pct = (avg_cat / avg_inc * 100) if avg_inc else 0
                lines.append(f"  • {friendly_category(cat)}: ~${avg_cat:,.2f}/month ({pct:.1f}% of income)")
            lines.append("")
            if savings_rate >= 20:
                lines.append(
                    f"Your savings rate of {savings_rate:.1f}% already meets the 20% benchmark — you're on solid ground. "
                    f"If you want to push further, look at your discretionary categories for additional wiggle room."
                )
            elif savings_rate > 0:
                gap = round(target_savings - current_monthly_savings, 2)
                lines.append(
                    f"To hit the 20% savings target, you'd need to save ~${target_savings:,.2f}/month. "
                    f"You're currently at ${current_monthly_savings:,.2f}/month — a gap of ${gap:,.2f}. "
                    f"Look at your discretionary categories first: small consistent cuts there add up fast."
                )
            else:
                lines.append(
                    f"You're running at a monthly deficit of ${abs(current_monthly_savings):,.2f}. "
                    f"Start with your largest non-essential categories — even reducing one by 20-30% would move the needle significantly."
                )
            _try_chart("plot_income_vs_expense")

        elif any(w in prompt_lower for w in ["compare", "versus", "vs ", "income vs", "income and expense"]):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            direction = "ahead" if net_savings >= 0 else "behind"
            lines.append(
                f"Income vs expenses — the core financial equation. "
                f"You've earned ${total_inc:,.2f} and spent ${total_exp:,.2f} over this period, "
                f"putting you ${abs(net_savings):,.2f} {direction} overall "
                f"({'a {:.1f}% savings rate'.format(savings_rate) if net_savings >= 0 else 'a deficit of {:.1f}%'.format(abs(savings_rate))})."
            )
            lines.append("\nMonth-by-month comparison:")
            idx = monthly_exp.index.union(monthly_inc.index)
            for dt in idx:
                inc_v = monthly_inc.get(dt, 0)
                exp_v = monthly_exp.get(dt, 0)
                month_net = inc_v - exp_v
                flag = " ✓" if month_net >= 0 else " ←"
                lines.append(f"  • {dt.strftime('%b %Y')}: Income ${inc_v:,.2f} | Expenses ${exp_v:,.2f} | Net ${month_net:+,.2f}{flag}")
            _try_chart("plot_income_vs_expense")

        elif any(w in prompt_lower for w in ["most", "highest", "top", "biggest", "maximum", "largest"]):
            if not top_cats.empty:
                top_pct = (top_cats.iloc[0] / total_exp * 100) if total_exp else 0
                top3_pct = sum((v / total_exp * 100) for v in top_cats.iloc[:3].values) if total_exp else 0
                lines.append(
                    f"Your biggest spending category is **{friendly_category(top_cats.index[0])}** at ${top_cats.iloc[0]:,.2f} "
                    f"({top_pct:.1f}% of everything you've spent). Your top 3 categories together account for {top3_pct:.0f}% of all spending:"
                )
                for cat, amt in top_cats.items():
                    pct = (amt / total_exp * 100) if total_exp else 0
                    lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
                if top_pct > 50:
                    lines.append(
                        f"\nWorth noting: {friendly_category(top_cats.index[0])} alone accounts for more than half your spending. "
                        f"This is often a fixed cost (like rent or loan payments), but it's worth reviewing if there's flexibility."
                    )
            _try_chart("plot_category_breakdown")
            _try_chart("plot_top_merchants")

        elif any(w in prompt_lower for w in ["least", "lowest", "smallest", "minimum", "less"]):
            bottom = all_cats.nsmallest(5)
            bottom_total = bottom.sum()
            bottom_pct = (bottom_total / total_exp * 100) if total_exp else 0
            lines.append(
                f"Your lowest spending categories — these collectively account for just {bottom_pct:.1f}% of your total expenses:"
            )
            for cat, amt in bottom.items():
                pct = (amt / total_exp * 100) if total_exp else 0
                lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
            lines.append(
                f"\nThese are minor cost centres. If you're looking to cut spending, the bigger levers are in your top categories."
            )
            _try_chart("plot_category_breakdown", top_n=10)

        elif any(w in prompt_lower for w in ["saving", "save", "savings", "net", "left over", "surplus"]):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            monthly_savings = net_savings / num_months if num_months else 0

            if net_savings > 0:
                if savings_rate >= 20:
                    verdict = (
                        f"Your savings rate of {savings_rate:.1f}% clears the recommended 20% benchmark — "
                        f"you're in strong financial shape."
                    )
                else:
                    verdict = (
                        f"Your savings rate of {savings_rate:.1f}% is positive, but below the standard 20% target. "
                        f"Closing that gap by even 5% per month would compound meaningfully over time."
                    )
                lines.append(
                    f"You're net positive — ${net_savings:,.2f} saved overall, averaging ${monthly_savings:,.2f}/month. "
                    + verdict
                )
            else:
                monthly_deficit = abs(monthly_savings)
                lines.append(
                    f"Your spending is currently outpacing your income by ${abs(net_savings):,.2f} total "
                    f"— roughly ${monthly_deficit:,.2f}/month in the red. "
                    f"The fastest path to reversing this is identifying your top 1-2 discretionary categories and cutting them back."
                )
            lines.append(f"\n  Total Income: ${total_inc:,.2f}")
            lines.append(f"  Total Expenses: ${total_exp:,.2f}")
            lines.append(f"  Net: ${net_savings:+,.2f} ({savings_rate:.1f}% savings rate)")
            if any(w in prompt_lower for w in ["trend", "pattern", "over time"]):
                lines.append(f"\nMonthly spending trend:")
                for dt, amt in monthly_exp.items():
                    lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f}")
            _try_chart("plot_income_vs_expense")
            _try_chart("plot_cumulative_flow")
            if any(w in prompt_lower for w in ["trend", "spend"]):
                _try_chart("plot_monthly_spending_trend")

        elif any(w in prompt_lower for w in ["trend", "over time", "pattern", "month by month", "monthly",
                                              "history", "progression", "growth"]):
            if not monthly_exp.empty and len(monthly_exp) >= 2:
                avg = monthly_exp.mean()
                highest_month = monthly_exp.idxmax().strftime("%b %Y")
                lowest_month = monthly_exp.idxmin().strftime("%b %Y")
                change = monthly_exp.iloc[-1] - monthly_exp.iloc[-2]
                direction = "increased" if change > 0 else "decreased"
                pct_change = (abs(change) / monthly_exp.iloc[-2] * 100) if monthly_exp.iloc[-2] else 0
                lines.append(
                    f"Your spending trend over {len(monthly_exp)} months: average ${avg:,.2f}/month, "
                    f"peaking in {highest_month} (${monthly_exp.max():,.2f}) and lowest in {lowest_month} "
                    f"(${monthly_exp.min():,.2f}). Most recently, spending {direction} by "
                    f"${abs(change):,.2f} ({pct_change:.1f}%) in {last_month_name}."
                )
                above_avg = [(dt.strftime("%b %Y"), amt) for dt, amt in monthly_exp.items() if amt > avg * 1.2]
                if above_avg:
                    spikes = ", ".join(f"{m} (${a:,.0f})" for m, a in above_avg)
                    lines.append(
                        f"\nMonths with notably elevated spending (20%+ above average): {spikes}. "
                        f"These spikes are worth digging into — they often point to one-off big purchases or seasonal patterns."
                    )
                lines.append("\nFull monthly breakdown:")
                for dt, amt in monthly_exp.items():
                    flag = " ←" if amt > avg * 1.2 else ""
                    lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f}{flag}")
            elif not monthly_exp.empty:
                lines.append(f"Only one month of data available — spending was ${monthly_exp.iloc[0]:,.2f}.")
            else:
                lines.append("No monthly spending data available yet.")
            _try_chart("plot_monthly_spending_trend")
            _try_chart("plot_category_trends")
            _try_chart("plot_weekly_pattern")

        elif any(w in prompt_lower for w in ["last month", "recent", "latest", "this month", "past month"]):
            overall_avg = monthly_exp.mean() if not monthly_exp.empty else 0
            vs_avg = last_month_amt - overall_avg
            vs_label = f"${abs(vs_avg):,.2f} {'above' if vs_avg > 0 else 'below'} your monthly average"
            lines.append(
                f"In {last_month_name}, you spent ${last_month_amt:,.2f} — {vs_label}."
            )
            last_month_dt = monthly_exp.index[-1] if not monthly_exp.empty else None
            if last_month_dt is not None:
                lm_txns = expenses[
                    (expenses["transaction_date"].dt.year == last_month_dt.year) &
                    (expenses["transaction_date"].dt.month == last_month_dt.month)
                ]
                lm_cats = lm_txns.groupby("transaction_category_detail")["transaction_amount"].sum().nlargest(5)
                if not lm_cats.empty:
                    top_lm_name = friendly_category(lm_cats.index[0])
                    top_lm_pct = (lm_cats.iloc[0] / last_month_amt * 100) if last_month_amt else 0
                    lines.append(
                        f"The biggest driver was **{top_lm_name}** at ${lm_cats.iloc[0]:,.2f} ({top_lm_pct:.1f}%). "
                        f"Here's the full category breakdown for {last_month_name}:"
                    )
                    for cat, amt in lm_cats.items():
                        pct = (amt / last_month_amt * 100) if last_month_amt else 0
                        lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
            _try_chart("plot_monthly_spending_trend")

        elif any(w in prompt_lower for w in ["breakdown", "category", "categories", "where", "split",
                                              "distribution", "pie", "donut"]):
            top3_pct = sum((v / total_exp * 100) for v in list(all_cats.values())[:3]) if total_exp else 0
            top_name = friendly_category(all_cats.index[0]) if not all_cats.empty else "N/A"
            lines.append(
                f"Here's where your money goes. Your top 3 categories account for {top3_pct:.0f}% of all spending, "
                f"with **{top_name}** leading the pack:"
            )
            for cat, amt in all_cats.items():
                pct = (amt / total_exp * 100) if total_exp else 0
                lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
            _try_chart("plot_category_breakdown", top_n=10)
            _try_chart("plot_category_trends")

        elif any(w in prompt_lower for w in ["income", "earn", "salary", "credit", "received", "paid me"]):
            inc_cats = income.groupby("transaction_category_detail")["transaction_amount"].sum().abs().nlargest(5)
            monthly_avg_inc = round(total_inc / num_months, 2) if num_months else 0
            lines.append(
                f"Your total income across this period is ${total_inc:,.2f} — averaging ${monthly_avg_inc:,.2f}/month."
            )
            if not inc_cats.empty:
                lines.append("Here's how your income breaks down by source:")
                for cat, amt in inc_cats.items():
                    pct = (amt / total_inc * 100) if total_inc else 0
                    lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            lines.append(
                f"\nOf that income, you've retained ${net_savings:,.2f} as savings "
                f"({'a {:.1f}% savings rate — above the 20% target'.format(savings_rate) if savings_rate >= 20 else 'a {:.1f}% savings rate — below the recommended 20%'.format(savings_rate) if savings_rate > 0 else 'deficit — spending exceeds income'})."
            )
            _try_chart("plot_income_vs_expense")

        elif any(w in prompt_lower for w in ["spend", "expens", "expenditure", "cost", "bought", "paid",
                                              "purchase", "pay", "bill"]):
            monthly_avg = round(total_exp / num_months, 2) if num_months else 0
            top_name = friendly_category(top_cats.index[0]) if not top_cats.empty else "N/A"
            top_pct = (top_cats.iloc[0] / total_exp * 100) if (not top_cats.empty and total_exp) else 0
            lines.append(
                f"You've spent ${total_exp:,.2f} across {len(expenses):,} transactions — "
                f"averaging ${monthly_avg:,.2f}/month. Your biggest cost driver is **{top_name}** "
                f"at {top_pct:.1f}% of total spend."
            )
            lines.append(f"\nTop spending categories:")
            for cat, amt in top_cats.items():
                pct = (amt / total_exp * 100) if total_exp else 0
                lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
            lines.append(f"\nMost recent month ({last_month_name}): ${last_month_amt:,.2f}")
            _try_chart("plot_category_breakdown")
            _try_chart("plot_expense_distribution")

        elif any(w in prompt_lower for w in ["merchant", "store", "shop", "vendor", "where do i", "places"]):
            top_merchants = expenses.groupby("merchant_name")["transaction_amount"].sum().nlargest(10)
            if not top_merchants.empty:
                top_m_name = top_merchants.index[0]
                top_m_amt = top_merchants.iloc[0]
                top_m_pct = (top_m_amt / total_exp * 100) if total_exp else 0
                top_m_count = len(expenses[expenses["merchant_name"] == top_m_name])
                lines.append(
                    f"Your top merchant is **{top_m_name}** — ${top_m_amt:,.2f} across {top_m_count} visits "
                    f"({top_m_pct:.1f}% of all spending). Here are your top 10 by spend:"
                )
                for merch, amt in top_merchants.items():
                    pct = (amt / total_exp * 100) if total_exp else 0
                    lines.append(f"  • {merch}: ${amt:,.2f} ({pct:.1f}%)")
            else:
                lines.append("No merchant data found in your transactions.")
            _try_chart("plot_top_merchants")
            _try_chart("plot_category_breakdown")

        elif any(w in prompt_lower for w in ["average", "mean", "typical", "normal", "usually"]):
            avg_monthly_exp = round(monthly_exp.mean(), 2) if not monthly_exp.empty else 0
            avg_txn = round(expenses["transaction_amount"].mean(), 2) if not expenses.empty else 0
            avg_monthly_inc = round(total_inc / num_months, 2) if num_months else 0
            txns_per_month = round(len(expenses) / num_months, 1) if num_months else 0
            lines.append(
                f"On a typical month, you spend ${avg_monthly_exp:,.2f} and earn ${avg_monthly_inc:,.2f} "
                f"— that's {round(txns_per_month)} transactions averaging ${avg_txn:,.2f} each."
            )
            savings_rate = ((avg_monthly_inc - avg_monthly_exp) / avg_monthly_inc * 100) if avg_monthly_inc else 0
            if savings_rate >= 20:
                lines.append(
                    f"Your typical monthly savings rate of {savings_rate:.1f}% beats the 20% benchmark — solid."
                )
            elif savings_rate > 0:
                lines.append(
                    f"Your typical monthly savings rate is {savings_rate:.1f}% — below the 20% target. "
                    f"Cutting discretionary spend by just ${round(avg_monthly_inc * 0.20 - (avg_monthly_inc - avg_monthly_exp), 0):,.0f}/month would close that gap."
                )
            else:
                lines.append(
                    f"On average, your spending exceeds income by ${abs(avg_monthly_exp - avg_monthly_inc):,.2f}/month — worth addressing."
                )
            _try_chart("plot_monthly_spending_trend")

        elif any(w in prompt_lower for w in ["graph", "chart", "plot", "visual", "show me", "display",
                                              "draw", "diagram"]):
            # ── Context-aware chart selection ──
            # Look at conversation history to pick the most relevant chart(s)
            context_topic = ""
            if conversation:
                recent_text = " ".join(
                    m.get("content", "") for m in conversation[-6:]
                ).lower()
                # Detect what the user has been discussing
                if any(w in recent_text for w in ["trend", "monthly", "over time", "month by month"]):
                    context_topic = "trend"
                elif any(w in recent_text for w in ["category", "breakdown", "where", "split"]):
                    context_topic = "category"
                elif any(w in recent_text for w in ["merchant", "store", "shop", "vendor"]):
                    context_topic = "merchant"
                elif any(w in recent_text for w in ["saving", "income", "expense", "earn"]):
                    context_topic = "savings"
                elif any(w in recent_text for w in ["budget", "plan", "recommend"]):
                    context_topic = "budget"
                elif any(w in recent_text for w in ["week", "day", "daily", "pattern"]):
                    context_topic = "weekly"

            # Also check the prompt itself for hints
            if any(w in prompt_lower for w in ["trend", "monthly", "over time"]):
                context_topic = "trend"
            elif any(w in prompt_lower for w in ["category", "breakdown", "pie", "donut"]):
                context_topic = "category"
            elif any(w in prompt_lower for w in ["merchant", "store", "shop"]):
                context_topic = "merchant"
            elif any(w in prompt_lower for w in ["saving", "income", "expense"]):
                context_topic = "savings"
            elif any(w in prompt_lower for w in ["week", "day", "daily"]):
                context_topic = "weekly"
            elif any(w in prompt_lower for w in ["distribution", "histogram", "size"]):
                context_topic = "distribution"
            elif any(w in prompt_lower for w in ["cumulative", "flow", "trajectory"]):
                context_topic = "cumulative"
            elif any(w in prompt_lower for w in ["heatmap", "heat map", "calendar"]):
                context_topic = "heatmap"
            elif any(w in prompt_lower for w in ["ratio", "rate"]):
                context_topic = "ratio"
            elif any(w in prompt_lower for w in ["scatter", "timeline", "transaction"]):
                context_topic = "scatter"
            elif any(w in prompt_lower for w in ["deep", "drill", "detail"]):
                context_topic = "deepdive"

            if context_topic == "trend":
                lines.append("Here's your spending trend — look for the direction and any months that spike:")
                _try_chart("plot_monthly_spending_trend")
                _try_chart("plot_category_trends")
            elif context_topic == "category":
                lines.append("Here's the category breakdown — the biggest slice is where to start if you want to cut:")
                _try_chart("plot_category_breakdown")
                _try_chart("plot_category_trends")
            elif context_topic == "merchant":
                lines.append("Here's your merchant ranking — the bars show where the most money actually goes:")
                _try_chart("plot_top_merchants")
            elif context_topic == "savings":
                lines.append("Here's the income vs expense picture — green above means you're saving, red above means overspending:")
                _try_chart("plot_income_vs_expense")
                _try_chart("plot_cumulative_flow")
            elif context_topic == "budget":
                lines.append("Here's your budgeting view — compare income and expenses side by side:")
                _try_chart("plot_income_vs_expense")
                _try_chart("plot_category_breakdown")
            elif context_topic == "weekly":
                lines.append("Here's your weekly spending pattern — you'll see which days are your most expensive:")
                _try_chart("plot_weekly_pattern")
            elif context_topic == "distribution":
                lines.append("Here's the transaction distribution — it shows whether your spend is driven by many small purchases or a few big ones:")
                _try_chart("plot_expense_distribution")
            elif context_topic == "cumulative":
                lines.append("Here's the cumulative flow — the gap between the lines tells you if you're getting ahead or falling behind:")
                _try_chart("plot_cumulative_flow")
            elif context_topic == "heatmap":
                lines.append("Here's the spending heatmap — darker cells mean heavier spending in that category/month combination:")
                _try_chart("plot_monthly_heatmap")
            elif context_topic == "ratio":
                lines.append("Here's your savings ratio over time — the dashed line is the 20% target:")
                _try_chart("plot_savings_ratio")
            elif context_topic == "scatter":
                lines.append("Here's every transaction plotted by date and amount — outliers jump out immediately:")
                _try_chart("plot_transaction_scatter")
            elif context_topic == "deepdive":
                lines.append("Here's a deep dive into your top category — it breaks down by merchant to show where the money actually goes:")
                _try_chart("plot_category_deepdive")
            else:
                # No context — ask the user what they want instead of dumping all charts
                lines.append("I can show you several types of charts. What would you like to see?\n")
                lines.append("  • **Spending trend** — how your spending changes month to month")
                lines.append("  • **Category breakdown** — where your money goes (donut chart)")
                lines.append("  • **Income vs expenses** — are you saving or overspending?")
                lines.append("  • **Top merchants** — which stores you spend the most at")
                lines.append("  • **Weekly pattern** — which days you spend the most")
                lines.append("  • **Category trends** — how categories shift over time")
                lines.append("  • **Transaction distribution** — small vs large purchases")
                lines.append("  • **Cumulative flow** — your total income vs expenses over time")
                lines.append("  • **Spending heatmap** — monthly heat map of categories")
                lines.append("  • **Savings ratio** — monthly savings rate over time")
                lines.append("  • **Transaction timeline** — every transaction plotted by date & amount")
                lines.append("  • **Category deep-dive** — detailed look at your biggest category")
                lines.append("\nJust tell me which one interests you, or describe what you'd like to understand!")

        elif any(w in prompt_lower for w in ["weekday", "weekly", "day of week", "day of the week", "which day",
                                              "what day", "weekend"]):
            _try_chart("plot_weekly_pattern")
            lines.append("Your weekly spending pattern is charted below — it shows average daily spend by weekday. "
                         "Look for weekend vs weekday differences.")

        elif any(w in prompt_lower for w in ["heatmap", "heat map", "calendar", "matrix"]):
            _try_chart("plot_monthly_heatmap")
            lines.append("Here's your spending heatmap — each cell shows category spend for a given month. "
                         "Darker = heavier spending.")

        elif any(w in prompt_lower for w in ["health", "score", "how am i doing", "financial health",
                                              "overview", "status", "check"]):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            monthly_savings = net_savings / num_months if num_months else 0
            if savings_rate >= 20:
                verdict = f"Strong — you're saving {savings_rate:.1f}% of your income, above the 20% benchmark."
            elif savings_rate > 0:
                verdict = f"Moderate — {savings_rate:.1f}% savings rate. There's room to improve toward the 20% target."
            else:
                verdict = f"Needs attention — you're running a deficit of {abs(savings_rate):.1f}%."
            lines.append(
                f"**Financial health check for {user_name}:** {verdict}\n"
            )
            lines.append(f"  • Monthly income (avg): ${round(total_inc / num_months, 2):,.2f}")
            lines.append(f"  • Monthly expenses (avg): ${round(total_exp / num_months, 2):,.2f}")
            lines.append(f"  • Monthly savings (avg): ${round(monthly_savings, 2):,.2f}")
            lines.append(f"  • Savings rate: {savings_rate:.1f}% (target: 20%)")
            lines.append(f"\n  Top expense: {friendly_category(top_cats.index[0])} (${top_cats.iloc[0]:,.2f})")
            _try_chart("plot_savings_ratio")
            _try_chart("plot_income_vs_expense")

        elif any(w in prompt_lower for w in ["report", "summary", "everything", "full",
                                              "complete", "all", "dashboard"]):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            lines.append(
                f"Full financial report for {user_name}. Over {num_months} months, you earned "
                f"${total_inc:,.2f} and spent ${total_exp:,.2f} — net {'savings' if net_savings >= 0 else 'deficit'} "
                f"of ${abs(net_savings):,.2f} ({savings_rate:.1f}% savings rate).\n"
            )
            lines.append(f"**Monthly averages:**")
            lines.append(f"  • Income: ${round(total_inc / num_months, 2):,.2f}")
            lines.append(f"  • Expenses: ${round(total_exp / num_months, 2):,.2f}")
            lines.append(f"\n**Top spending categories:**")
            for cat, amt in top_cats.items():
                pct = (amt / total_exp * 100) if total_exp else 0
                lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
            top_merchants = expenses.groupby("merchant_name")["transaction_amount"].sum().nlargest(5)
            if not top_merchants.empty:
                lines.append(f"\n**Top merchants:**")
                for merch, amt in top_merchants.items():
                    lines.append(f"  • {merch}: ${amt:,.2f}")
            _try_chart("plot_category_breakdown")
            _try_chart("plot_monthly_spending_trend")
            _try_chart("plot_income_vs_expense")
            _try_chart("plot_top_merchants")
            _try_chart("plot_cumulative_flow")
            _try_chart("plot_category_trends")
            _try_chart("plot_expense_distribution")
            _try_chart("plot_weekly_pattern")

        elif any(w in prompt_lower for w in ["scatter", "timeline", "every transaction", "all transactions"]):
            _try_chart("plot_transaction_scatter")
            lines.append("Every transaction plotted by date and amount — outliers and clusters are easy to spot in this view.")

        elif any(w in prompt_lower for w in ["cumulative", "flow", "trajectory", "net position"]):
            _try_chart("plot_cumulative_flow")
            lines.append("Your cumulative cash flow chart shows total income vs total expenses over time. "
                         "A widening gap with income on top means you're building savings; the opposite signals overspending.")

        elif any(w in prompt_lower for w in ["how much", "total", "amount", "sum"]):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            lines.append(
                f"Your totals: ${total_exp:,.2f} spent, ${total_inc:,.2f} earned — a net of ${net_savings:+,.2f} "
                f"across {len(user_df):,} transactions ({savings_rate:.1f}% savings rate)."
            )
            _try_chart("plot_income_vs_expense")

        elif any(w in prompt_lower for w in ["day", "date", "when", "time"]):
            _try_chart("plot_weekly_pattern")
            _try_chart("plot_transaction_scatter")
            lines.append("Here's when your spending happens — the weekly pattern shows average spend by day, "
                         "and the scatter plot shows every transaction by date.")

        elif any(w in prompt_lower for w in ["transaction", "count", "number", "how many"]):
            txns_per_month = round(len(expenses) / num_months, 1) if num_months else 0
            avg_txn = round(expenses["transaction_amount"].mean(), 2) if not expenses.empty else 0
            lines.append(
                f"You've made {len(user_df):,} total transactions ({len(expenses):,} expenses, "
                f"{len(income):,} income entries) — averaging {txns_per_month:.0f} transactions/month "
                f"at ${avg_txn:,.2f} each."
            )
            _try_chart("plot_expense_distribution")

        elif any(w in prompt_lower for w in ["hello", "hi ", "hey", "greet", "good morning", "good evening"]):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            lines.append(
                f"Hey {user_name}! I'm your financial analyst — I have your complete transaction history loaded "
                f"and ready to go. Quick snapshot: you've spent ${total_exp:,.2f} and earned ${total_inc:,.2f} "
                f"({'saving {:.1f}%'.format(savings_rate) if savings_rate > 0 else 'currently in deficit'}). "
                f"What would you like to dig into?"
            )

        elif any(w in prompt_lower for w in ["thank", "thanks", "appreciate", "great", "awesome", "perfect",
                                              "nice", "good job", "well done"]):
            lines.append(
                f"You're welcome, {user_name}! If you want to keep going — you could ask about your "
                f"spending trends, check your savings rate, or deep-dive into a specific category or merchant. "
                f"I'm here whenever you need."
            )

        elif any(w in prompt_lower for w in ["help", "what can you", "what do you", "options", "commands"]):
            lines.append(f"Here's what I can help you with, {user_name}:\n")
            lines.append("  • **Spending analysis** — \"What did I spend the most on?\"")
            lines.append("  • **Trends** — \"Show my spending trends over time\"")
            lines.append("  • **Savings** — \"Am I saving money?\"")
            lines.append("  • **Categories** — \"Break down my expenses by category\"")
            lines.append("  • **Merchants** — \"Where do I shop the most?\"")
            lines.append("  • **Income** — \"How much did I earn?\"")
            lines.append("  • **Comparisons** — \"Compare my income and expenses\"")
            lines.append("  • **Budget** — \"How should I budget next month?\"")
            lines.append("  • **Charts** — \"Show me a spending trend chart\"")
            lines.append("  • **Full report** — \"Give me a complete financial report\"")
            lines.append("  • **Specific queries** — Ask about any category or merchant by name")
            lines.append("\nJust ask anything about your finances — I'll pull the data and show you.")

        else:
            # SMART CATCH-ALL: analyze what's most relevant
            # Check conversation context for follow-ups
            if conversation and len(conversation) >= 2:
                last_assistant = ""
                for msg in reversed(conversation):
                    if msg.get("role") == "assistant":
                        last_assistant = msg.get("content", "").lower()
                        break
                # If last response was about a topic, continue that topic
                if "category" in last_assistant or "breakdown" in last_assistant:
                    lines.append("Here's more detail on your spending categories:\n")
                    for cat, amt in all_cats.items():
                        pct = (amt / total_exp * 100) if total_exp else 0
                        lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
                    _try_chart("plot_category_breakdown", top_n=10)
                elif "trend" in last_assistant or "monthly" in last_assistant:
                    lines.append("Here's the extended spending trend:\n")
                    for dt, amt in monthly_exp.items():
                        flag = " ← above avg" if amt > monthly_exp.mean() * 1.2 else ""
                        lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f}{flag}")
                    _try_chart("plot_monthly_spending_trend", months=12)
                elif "saving" in last_assistant or "income" in last_assistant:
                    savings_rate = (net_savings / total_inc * 100) if total_inc else 0
                    lines.append(
                        f"More on your income and savings: you're at a {savings_rate:.1f}% savings rate, "
                        f"averaging ${round(net_savings / num_months, 2):,.2f}/month in net savings."
                    )
                    _try_chart("plot_income_vs_expense", months=12)
                else:
                    self._fallback_generic(lines, user_name, user_df, expenses, income,
                                           total_exp, total_inc, net_savings, top_cats, num_months)
                    _try_chart("plot_category_breakdown")
                    _try_chart("plot_monthly_spending_trend")
            else:
                self._fallback_generic(lines, user_name, user_df, expenses, income,
                                       total_exp, total_inc, net_savings, top_cats, num_months)
                _try_chart("plot_category_breakdown")
                _try_chart("plot_monthly_spending_trend")

        if chart_paths:
            lines.append(f"\n📊 {len(chart_paths)} chart(s) generated for you below.")

        return "\n".join(lines), chart_paths

    @staticmethod
    def _fallback_generic(lines, user_name, user_df, expenses, income,
                          total_exp, total_inc, net_savings, top_cats, num_months):
        """Generic comprehensive response when no specific intent is matched."""
        savings_rate = (net_savings / total_inc * 100) if total_inc else 0
        if savings_rate >= 20:
            verdict = f"In good shape — saving {savings_rate:.1f}% of income."
        elif savings_rate > 0:
            verdict = f"Saving {savings_rate:.1f}%, but below the 20% target."
        else:
            verdict = f"Running a deficit of {abs(savings_rate):.1f}% — spending exceeds income."

        lines.append(f"Here's your financial snapshot, {user_name} — **{verdict}**\n")
        lines.append(f"  • Transactions: {len(user_df):,}")
        lines.append(f"  • Total expenses: ${total_exp:,.2f}")
        lines.append(f"  • Total income: ${total_inc:,.2f}")
        lines.append(f"  • Net: ${net_savings:+,.2f} ({savings_rate:.1f}% savings rate)")
        lines.append(f"\nTop spending categories:")
        for cat, amt in top_cats.items():
            pct = (amt / total_exp * 100) if total_exp else 0
            lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
        lines.append(
            f"\nAsk me about any category, merchant, trends, savings, or budgeting — "
            f"or just say 'show me a chart' and I'll pick the most relevant one."
        )

    def _explain_chart(self, chart_info, user_df: pd.DataFrame) -> str:
        """Generate a concise data-driven explanation for a chart.

        chart_info can be:
          - a str (bare path, legacy)
          - a dict with keys: path, fn (function name), kwargs (chart params)
        """
        if isinstance(chart_info, dict):
            fname = os.path.basename(chart_info.get("path", "")).lower()
            fn_name = chart_info.get("fn", "")
            kwargs = chart_info.get("kwargs", {})
        else:
            fname = os.path.basename(chart_info).lower()
            fn_name = ""
            kwargs = {}

        all_expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]

        # ── Apply the same time filter the chart function used ──
        def _filter(df, months):
            cutoff = df["transaction_date"].max() - pd.DateOffset(months=months)
            return df[df["transaction_date"] >= cutoff]

        period_map = {"last_month": 1, "last_3_months": 3, "last_6_months": 6, "all_time": 120}

        if "spending_trend" in fname or fn_name == "plot_monthly_spending_trend":
            months = kwargs.get("months", 6)
            expenses = _filter(all_expenses, months)
            monthly = expenses.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
            if not monthly.empty and len(monthly) >= 2:
                avg = monthly.mean()
                high_m = monthly.idxmax().strftime("%b %Y")
                low_m = monthly.idxmin().strftime("%b %Y")
                change = monthly.iloc[-1] - monthly.iloc[-2]
                direction = "up" if change > 0 else "down"
                return (f"This line chart tracks your monthly spending over {len(monthly)} months. "
                        f"Your average is ${avg:,.0f}/month, peaking in {high_m} (${monthly.max():,.0f}) "
                        f"and lowest in {low_m} (${monthly.min():,.0f}). "
                        f"Last month went {direction} by ${abs(change):,.0f}. "
                        f"The dashed orange line shows the 3-month rolling average to smooth out short-term fluctuations.")
            return "This line chart shows your monthly spending over time with a rolling average overlay."

        elif "category_breakdown" in fname or fn_name == "plot_category_breakdown":
            period = kwargs.get("period", "last_3_months")
            months = period_map.get(period, 3)
            expenses = _filter(all_expenses, months)
            period_label = period.replace("_", " ").replace("last ", "last ")
            cat_totals = expenses.groupby("transaction_category_detail")["transaction_amount"].sum().nlargest(5)
            total = expenses["transaction_amount"].sum()
            if not cat_totals.empty:
                top_name = friendly_category(cat_totals.index[0])
                top_pct = (cat_totals.iloc[0] / total * 100) if total else 0
                top3_pct = sum(v / total * 100 for v in cat_totals.iloc[:3].values) if total else 0
                return (f"This donut chart breaks down your spending over the {period_label}. "
                        f"The centre shows the total for that period (${total:,.0f}). "
                        f"{top_name} dominates at {top_pct:.0f}%, and your top 3 categories "
                        f"account for {top3_pct:.0f}% of spending. "
                        f"Smaller slices are grouped as 'Other'.")
            return "This donut chart shows how your spending is distributed across categories."

        elif "income_vs_expense" in fname or fn_name == "plot_income_vs_expense":
            months = kwargs.get("months", 6)
            filtered = _filter(user_df, months)
            f_exp = filtered[filtered["transaction_amount"] > 0]["transaction_amount"].sum()
            f_inc = abs(filtered[filtered["transaction_amount"] < 0]["transaction_amount"].sum())
            net = f_inc - f_exp
            return (f"Green bars represent monthly income, red bars represent expenses (last {months} months). "
                    f"In this period you earned ${f_inc:,.0f} and spent ${f_exp:,.0f}. "
                    f"{'The orange diamond line shows your net savings each month — months above the grey dashed line mean you saved money.' if net > 0 else 'The orange diamond line shows net savings — points below the grey dashed line indicate months where spending exceeded income.'}")

        elif "top_merchants" in fname or fn_name == "plot_top_merchants":
            months = kwargs.get("months", 12)
            expenses = _filter(all_expenses, months)
            merch = expenses.groupby("merchant_name")["transaction_amount"].sum().nlargest(3)
            if not merch.empty:
                top_name = merch.index[0]
                top_amt = merch.iloc[0]
                return (f"This horizontal bar chart ranks your merchants by total spend (last {months} months). "
                        f"Your #1 merchant is {top_name} at ${top_amt:,.0f}. "
                        f"Each bar shows the dollar amount and percentage of your total spending. "
                        f"Longer bars mean more money spent at that merchant.")
            return "This chart ranks your merchants by total amount spent."

        elif "weekly_pattern" in fname or fn_name == "plot_weekly_pattern":
            months = kwargs.get("months", 6)
            expenses = _filter(all_expenses, months).copy()
            expenses["dow"] = expenses["transaction_date"].dt.day_name()
            day_avg = expenses.groupby("dow")["transaction_amount"].mean()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_avg = day_avg.reindex(day_order).fillna(0)
            peak_day = day_avg.idxmax()
            low_day = day_avg.idxmin()
            return (f"This chart shows your average spend per day of the week (last {months} months). "
                    f"Blue bars are weekdays, green bars are weekends. "
                    f"You tend to spend the most on {peak_day}s (avg ${day_avg.max():,.0f}) "
                    f"and least on {low_day}s (avg ${day_avg.min():,.0f}). "
                    f"The transaction count below each bar shows how many purchases you typically make that day.")

        elif "category_trends" in fname or fn_name == "plot_category_trends":
            months = kwargs.get("months", 6)
            return (f"This stacked area chart shows how your top spending categories change month over month (last {months} months). "
                    "Each coloured band represents a category — wider bands mean higher spending. "
                    "Look for categories that are growing over time or sudden spikes that may need attention.")

        elif "expense_distribution" in fname or fn_name == "plot_expense_distribution":
            amounts = all_expenses["transaction_amount"]
            median = amounts.median()
            mean = amounts.mean()
            return (f"This histogram shows how your transactions are spread across different amount ranges. "
                    f"Your median transaction is ${median:,.0f} (red dashed line) and mean is ${mean:,.0f} (blue dashed line). "
                    f"{'Most of your transactions are smaller amounts with a few large outliers.' if mean > median * 1.3 else 'Your transactions are fairly evenly distributed.'} "
                    f"Warmer colours indicate higher amounts.")

        elif "cumulative_flow" in fname or fn_name == "plot_cumulative_flow":
            months = kwargs.get("months", 12)
            filtered = _filter(user_df, months)
            f_exp = filtered[filtered["transaction_amount"] > 0]["transaction_amount"].sum()
            f_inc = abs(filtered[filtered["transaction_amount"] < 0]["transaction_amount"].sum())
            net = f_inc - f_exp
            status = "ahead" if net > 0 else "behind"
            return (f"This chart plots your running total of income (green) vs expenses (red) over the last {months} months. "
                    f"The shaded gap between the lines shows your net position — "
                    f"{'green shading means you\'re earning more than spending' if net > 0 else 'red shading means spending is outpacing income'}. "
                    f"You're currently ${abs(net):,.0f} {status} overall.")

        elif "monthly_heatmap" in fname or fn_name == "plot_monthly_heatmap":
            months = kwargs.get("months", 6)
            return (f"This heatmap shows spending intensity across your top categories over {months} months. "
                    "Brighter purple cells indicate higher spending. "
                    "Scan across rows to see which categories are consistently high, "
                    "and look for bright spots that indicate spending spikes.")

        elif "savings_ratio" in fname or fn_name == "plot_savings_ratio":
            months = kwargs.get("months", 8)
            filtered = _filter(user_df, months)
            f_inc = abs(filtered[filtered["transaction_amount"] < 0]["transaction_amount"].sum())
            f_exp = filtered[filtered["transaction_amount"] > 0]["transaction_amount"].sum()
            avg_rate = ((f_inc - f_exp) / f_inc * 100) if f_inc else 0
            return (f"This chart shows your monthly savings rate over the last {months} months. "
                    f"Green bars mean you saved money that month; red bars mean you overspent. "
                    f"The dashed orange line marks the recommended 20% savings target. "
                    f"Your average savings rate is {avg_rate:.1f}%.")

        elif "transaction_scatter" in fname or fn_name == "plot_transaction_scatter":
            months = kwargs.get("months", 6)
            expenses = _filter(all_expenses, months)
            return (f"Each dot represents a single transaction over the last {months} months, "
                    f"plotted by date and amount. Colours indicate categories. "
                    f"Higher dots are bigger purchases. Look for clusters (frequent spending) "
                    f"or outlier dots (unusual large purchases). You had {len(expenses)} expense transactions in this period.")

        elif "category_deepdive" in fname or fn_name == "plot_category_deepdive":
            cat = kwargs.get("category", "")
            if cat:
                cat_label = friendly_category(cat)
            else:
                cat_label = friendly_category(all_expenses.groupby("transaction_category_detail")["transaction_amount"].sum().idxmax()) if not all_expenses.empty else "your top category"
            return (f"This chart drills into **{cat_label}** spending, breaking it down by merchant. "
                    f"Each bar shows how much you spent at that merchant within this category, plus the number of visits. "
                    f"This helps identify which specific merchants are driving your spending in this category.")

        return "This chart visualizes your financial data."

    # ══════════════════════════════════════════════════════════════
    #  run()  —  the 4-stage pipeline
    # ══════════════════════════════════════════════════════════════

    def run(self, user_id: str, prompt: str) -> dict:
        t0 = time.time()

        # ── Stage 1: Input & User Data Fetch ──

        user_df = self._user_df(user_id)
        if user_df is None:
            return {
                "user_name": None,
                "response": f"User '{user_id}' not found. Please provide a valid user ID.",
                "data_summary": {},
                "visualizations": [],
                "cache_hit": False,
                "latency_ms": round((time.time() - t0) * 1000),
                "guardrail_flags": ["invalid_user_id"],
            }

        user_name = user_df["user_name"].iloc[0]

        # Input guardrails
        gr = run_input_guardrails(prompt, user_id, self._all_user_ids)
        if gr["blocked"]:
            ms = round((time.time() - t0) * 1000)
            self.audit.log_request(user_id, prompt, gr["message"], ms, gr["flags"], False)
            return {
                "user_name": user_name,
                "response": gr["message"],
                "data_summary": {},
                "visualizations": [],
                "cache_hit": False,
                "latency_ms": ms,
                "guardrail_flags": gr["flags"],
            }

        processed_prompt = gr.get("prompt", prompt)
        flags = list(gr.get("flags", []))

        # Profile (cache hit / miss)
        cache_hit = False
        profile = self.cache.get_profile(user_id)
        if profile:
            cache_hit = True
        else:
            profile = self.cache.build_profile(user_id, user_df)

        # ── Stage 2: Context Assembly ──

        history = self.cache.get_query_history(user_id)
        viz_state = self.cache.get_viz_state(user_id) or {}
        conversation = self.cache.get_conversation(user_id)
        system_prompt = self._build_system_prompt(profile, history, viz_state=viz_state)
        system_prompt = enforce_token_budget(system_prompt, MAX_TOKENS_INPUT)

        # Build messages with conversation history for multi-turn context
        messages = [{"role": "system", "content": system_prompt}]
        # Include recent conversation turns (last 6 messages max)
        for msg in conversation[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": processed_prompt})

        # Record user turn in conversation
        self.cache.add_conversation_turn(user_id, "user", processed_prompt)

        # ── Stage 3: LLM Reasoning + Tool Dispatch ──

        data_summary = self._data_summary(user_df)
        llm_result = self.llm.chat(messages=messages, tools=TOOL_SCHEMAS)

        chart_paths: list[str] = []

        if llm_result.get("error"):
            response_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            flags.append("llm_fallback")
        else:
            response_text = llm_result.get("content", "") or ""
            tool_calls = llm_result.get("tool_calls") or []

            if tool_calls:
                chart_paths = self._execute_tool_calls(tool_calls, user_id)

            # Clean spurious <tool_call> markup from text
            response_text = self._clean_response(response_text)

            # If LLM only returned tool calls (no usable text), try follow-up or generate from data
            if not response_text.strip() and chart_paths:
                followup = messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                ]
                for tc in tool_calls:
                    followup.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_1"),
                        "content": json.dumps({"status": "success", "charts": chart_paths}),
                    })
                followup_result = self.llm.chat(messages=followup, tools=TOOL_SCHEMAS)
                if not followup_result.get("error"):
                    response_text = self._clean_response(
                        followup_result.get("content", "") or ""
                    )

            # Final fallback: generate text from raw data
            if not response_text.strip():
                response_text = self._summarise_from_data(user_df, processed_prompt, chart_paths)
                if not chart_paths:
                    flags.append("llm_empty_response")

        # ── Stage 4: Response Composition ──

        out_gr = run_output_guardrails(response_text, data_summary)
        flags.extend(out_gr.get("flags", []))

        # Update caches
        self.cache.add_query(
            user_id,
            processed_prompt,
            operation="llm_analysis",
            result_summary=response_text[:200],
        )
        # Record assistant turn in conversation
        self.cache.add_conversation_turn(user_id, "assistant", response_text[:500])
        # Update viz state if charts were generated
        if chart_paths:
            chart_type = chart_paths[-1].split("/")[-1].split("_")[0] if chart_paths else ""
            for fn_name in VIZ_FUNCTIONS:
                if fn_name in (chart_paths[-1] if chart_paths else ""):
                    chart_type = fn_name
                    break
            self.cache.set_viz_state(user_id, chart_type, {}, {})

        latency_ms = round((time.time() - t0) * 1000)
        self.audit.log_request(
            user_id, processed_prompt, response_text[:200],
            latency_ms, flags, cache_hit,
        )

        return {
            "user_name": user_name,
            "response": response_text,
            "data_summary": data_summary,
            "visualizations": [
                {"path": p, "explanation": self._explain_chart(p, user_df)}
                for p in chart_paths
            ] if chart_paths else [],
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "guardrail_flags": flags,
        }

    # ══════════════════════════════════════════════════════════════
    #  run_stream()  —  SSE streaming variant of run()
    # ══════════════════════════════════════════════════════════════

    def run_stream(self, user_id: str, prompt: str):
        """Generator that yields SSE-ready dicts for streaming responses.

        Event types:
          {"event": "meta",     "data": {user_name, cache_hit, ...}}
          {"event": "chunk",    "data": {"text": "..."}}
          {"event": "charts",   "data": {"visualizations": [...]}}
          {"event": "done",     "data": {latency_ms, guardrail_flags, data_summary}}
          {"event": "blocked",  "data": {response, guardrail_flags}}
        """
        t0 = time.time()

        # ── Stage 1: Input & User Data Fetch ──
        user_df = self._user_df(user_id)
        if user_df is None:
            yield {"event": "blocked", "data": {
                "response": f"User '{user_id}' not found.",
                "guardrail_flags": ["invalid_user_id"],
            }}
            return

        user_name = user_df["user_name"].iloc[0]

        gr = run_input_guardrails(prompt, user_id, self._all_user_ids)
        if gr["blocked"]:
            yield {"event": "blocked", "data": {
                "response": gr["message"],
                "guardrail_flags": gr["flags"],
            }}
            return

        processed_prompt = gr.get("prompt", prompt)
        flags = list(gr.get("flags", []))

        cache_hit = False
        profile = self.cache.get_profile(user_id)
        if profile:
            cache_hit = True
        else:
            profile = self.cache.build_profile(user_id, user_df)

        # ── Emit meta event ──
        yield {"event": "meta", "data": {
            "user_name": user_name,
            "cache_hit": cache_hit,
        }}

        # ── Stage 2: Context Assembly ──
        history = self.cache.get_query_history(user_id)
        viz_state = self.cache.get_viz_state(user_id) or {}
        conversation = self.cache.get_conversation(user_id)
        system_prompt = self._build_system_prompt(profile, history, viz_state=viz_state)
        system_prompt = enforce_token_budget(system_prompt, MAX_TOKENS_INPUT)

        # Build messages with conversation history for multi-turn context
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": processed_prompt})

        # Record user turn
        self.cache.add_conversation_turn(user_id, "user", processed_prompt)

        data_summary = self._data_summary(user_df)

        # ── Stage 3: LLM Streaming ──
        full_text = ""
        tool_calls = []
        llm_error = False
        chart_paths: list[str] = []

        for event in self.llm.chat_stream(messages=messages, tools=TOOL_SCHEMAS):
            if event["type"] == "chunk":
                full_text += event["content"]
                yield {"event": "chunk", "data": {"text": event["content"]}}
            elif event["type"] == "tool_calls":
                tool_calls = event["tool_calls"]
            elif event["type"] == "error":
                llm_error = True
                break

        if llm_error:
            # Fallback: stream the data-driven response word by word
            flags.append("llm_fallback")
            fallback_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            # Stream fallback text in small chunks to simulate typing
            words = fallback_text.split(" ")
            chunk = ""
            first_chunk = True
            for i, word in enumerate(words):
                chunk += (" " if chunk else "") + word
                if len(chunk) >= 30 or i == len(words) - 1:
                    # Add leading space between chunks so frontend concatenation preserves spacing
                    text_to_send = chunk if first_chunk else " " + chunk
                    yield {"event": "chunk", "data": {"text": text_to_send}}
                    chunk = ""
                    first_chunk = False
            full_text = fallback_text
        else:
            # Process tool calls if any
            if tool_calls:
                chart_paths = self._execute_tool_calls(tool_calls, user_id)

            # Clean response
            full_text = self._clean_response(full_text)

            # If only tool calls (no text), generate from data
            if not full_text.strip():
                full_text = self._summarise_from_data(user_df, processed_prompt, chart_paths)
                words = full_text.split(" ")
                chunk = ""
                first_chunk = True
                for i, word in enumerate(words):
                    chunk += (" " if chunk else "") + word
                    if len(chunk) >= 30 or i == len(words) - 1:
                        text_to_send = chunk if first_chunk else " " + chunk
                        yield {"event": "chunk", "data": {"text": text_to_send}}
                        chunk = ""
                        first_chunk = False
                if not chart_paths:
                    flags.append("llm_empty_response")

        # ── Emit charts with explanations ──
        if chart_paths:
            explained = []
            for p in chart_paths:
                if isinstance(p, dict):
                    # From fallback: already has path, fn, kwargs
                    explained.append({
                        "path": p["path"],
                        "explanation": self._explain_chart(p, user_df),
                    })
                else:
                    # From LLM tool calls: plain path string
                    explained.append({
                        "path": p,
                        "explanation": self._explain_chart(p, user_df),
                    })
            yield {"event": "charts", "data": {"visualizations": explained}}

        # ── Stage 4: Finalize ──
        out_gr = run_output_guardrails(full_text, data_summary)
        flags.extend(out_gr.get("flags", []))

        self.cache.add_query(
            user_id, processed_prompt,
            operation="llm_analysis",
            result_summary=full_text[:200],
        )
        # Record assistant turn in conversation
        self.cache.add_conversation_turn(user_id, "assistant", full_text[:500])
        # Update viz state if charts were generated
        if chart_paths:
            last_chart_item = chart_paths[-1]
            if isinstance(last_chart_item, dict):
                chart_type = last_chart_item.get("fn", "") or ""
                if not chart_type:
                    last_path = last_chart_item.get("path", "")
                    for fn_name in VIZ_FUNCTIONS:
                        if fn_name in last_path:
                            chart_type = fn_name
                            break
            else:
                chart_type = ""
                for fn_name in VIZ_FUNCTIONS:
                    if fn_name in last_chart_item:
                        chart_type = fn_name
                        break
            self.cache.set_viz_state(user_id, chart_type, {}, {})

        latency_ms = round((time.time() - t0) * 1000)
        self.audit.log_request(
            user_id, processed_prompt, full_text[:200],
            latency_ms, flags, cache_hit,
        )

        yield {"event": "done", "data": {
            "latency_ms": latency_ms,
            "guardrail_flags": flags,
            "data_summary": data_summary,
        }}
