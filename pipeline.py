"""Main TransactionRAGPipeline — the 4-stage orchestrator."""

import json
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

        self.cache = KVCache()
        self.llm = LLMClient()
        self.audit = AuditLogger()

    # ── helpers ──

    def _user_df(self, user_id: str) -> Optional[pd.DataFrame]:
        udf = self.df[self.df["user_id"] == user_id]
        return udf.copy() if not udf.empty else None

    def _build_system_prompt(self, profile: dict, query_history: list) -> str:
        system = (
            "You are a helpful financial analyst assistant. You analyze a user's "
            "transaction data and provide clear, accurate insights.\n\n"
            "RULES:\n"
            "- Only reference data that exists in the user's transactions.\n"
            "- If data is insufficient, say so clearly.\n"
            "- Use specific numbers from the data, not estimates.\n"
            "- When appropriate, call visualization tools to generate charts.\n"
            "- Keep responses concise and actionable.\n"
            "- Negative transaction amounts are INCOME (salary, refunds).\n"
            "- Positive transaction amounts are EXPENSES.\n\n"
            f"USER PROFILE:\n"
            f"- Name: {profile['user_name']}\n"
            f"- Data Range: {profile['date_range']['start']} to {profile['date_range']['end']}\n"
            f"- Total Transactions: {profile['total_transactions']}\n"
            f"- Avg Monthly Spend: ${profile['avg_monthly_spend']:,.2f}\n"
            f"- Avg Monthly Income: ${profile['avg_monthly_income']:,.2f}\n"
            f"- Top Expense Categories: {json.dumps(profile['top_expense_categories'])}\n\n"
            "AVAILABLE COLUMNS: user_id, user_name, transaction_date, "
            "transaction_amount, transaction_category_detail, merchant_name\n\n"
            "CATEGORIES follow the pattern: SUBCATEGORY_MAINCATEGORY "
            "(e.g., RENT_HOUSING, COFFEE_FOOD)\n"
        )

        if query_history:
            system += "\nPREVIOUS QUERIES BY THIS USER:\n"
            for i, q in enumerate(query_history[-3:], 1):
                system += f"{i}. Q: {q['prompt']}\n   Summary: {q['result_summary']}\n"

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
            "category_totals": (
                expenses.groupby("transaction_category_detail")["transaction_amount"]
                .sum()
                .to_dict()
            ),
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
        """Strip spurious <tool_call> XML that some models emit in the text field."""
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL)  # unclosed
        return text.strip()

    def _summarise_from_data(self, user_df: pd.DataFrame, prompt: str, chart_paths: list[str]) -> str:
        """Generate a short textual answer purely from the DataFrame when the LLM
        did not provide usable text (only tool calls)."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]
        prompt_lower = prompt.lower()

        lines: list[str] = []

        if any(w in prompt_lower for w in ["most", "top", "highest", "breakdown", "category", "where"]):
            top = (
                expenses.groupby("transaction_category_detail")["transaction_amount"]
                .sum().nlargest(5)
            )
            lines.append("Here are your top spending categories:")
            for cat, amt in top.items():
                lines.append(f"  • {cat}: ${amt:,.2f}")

        elif any(w in prompt_lower for w in ["trend", "over time", "month"]):
            monthly = (
                expenses.set_index("transaction_date")
                .resample("ME")["transaction_amount"].sum()
            )
            lines.append("Monthly spending summary:")
            for dt, amt in monthly.items():
                lines.append(f"  • {dt.strftime('%b %Y')}: ${amt:,.2f}")

        elif any(w in prompt_lower for w in ["saving", "save", "income", "expense", "financial"]):
            total_exp = expenses["transaction_amount"].sum()
            total_inc = abs(income["transaction_amount"].sum())
            lines.append(f"Total Income: ${total_inc:,.2f}")
            lines.append(f"Total Expenses: ${total_exp:,.2f}")
            lines.append(f"Net Savings: ${total_inc - total_exp:,.2f}")

        else:
            lines.append(f"Total transactions: {len(user_df)}")
            lines.append(f"Total expenses: ${expenses['transaction_amount'].sum():,.2f}")
            lines.append(f"Total income: ${abs(income['transaction_amount'].sum()):,.2f}")

        if chart_paths:
            lines.append(f"\n{len(chart_paths)} chart(s) have been generated for you.")
        return "\n".join(lines)

        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]
        top_cats = (
            expenses.groupby("transaction_category_detail")["transaction_amount"]
            .sum()
            .nlargest(3)
        )

        lines = [
            "I'm currently unable to reach the AI service, but here's a summary from your data:\n",
            f"• Total Transactions: {len(user_df)}",
            f"• Total Expenses: ${expenses['transaction_amount'].sum():,.2f}",
            f"• Total Income: ${abs(income['transaction_amount'].sum()):,.2f}",
            "• Top Spending Categories:",
        ]
        for cat, amt in top_cats.items():
            lines.append(f"  – {cat}: ${amt:,.2f}")
        return "\n".join(lines)

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
        gr = run_input_guardrails(prompt, user_id)
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
        system_prompt = self._build_system_prompt(profile, history)
        system_prompt = enforce_token_budget(system_prompt, MAX_TOKENS_INPUT)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": processed_prompt},
        ]

        # ── Stage 3: LLM Reasoning + Tool Dispatch ──

        data_summary = self._data_summary(user_df)
        llm_result = self.llm.chat(messages=messages, tools=TOOL_SCHEMAS)

        chart_paths: list[str] = []

        if llm_result.get("error"):
            response_text = self._fallback_response(user_df)
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
                followup_result = self.llm.chat(messages=followup)
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

        latency_ms = round((time.time() - t0) * 1000)
        self.audit.log_request(
            user_id, processed_prompt, response_text[:200],
            latency_ms, flags, cache_hit,
        )

        return {
            "user_name": user_name,
            "response": response_text,
            "data_summary": data_summary,
            "visualizations": chart_paths,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "guardrail_flags": flags,
        }
