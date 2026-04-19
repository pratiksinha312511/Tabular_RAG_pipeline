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
            "You are Vola AI, the intelligent financial assistant built into the Vola Finance platform. "
            "Vola is a membership-based fintech app that helps over 1 million members take control of their "
            "finances through instant cash advances (up to $500, no interest, no credit check), credit building "
            "(rent reporting, CreditMap dashboard), expense tracking with automatic categorization, budget "
            "planning with smart alerts, and subscription management — all in one app.\n\n"

            "You have direct, live access to this member's complete bank and card transaction history. "
            "You are not a generic chatbot — you are a sharp, data-driven financial advisor who speaks plainly, "
            "thinks in benchmarks, and always leads with the most useful insight.\n\n"

            "Your character: Think of yourself like a friend who's got the member's back — knowledgeable, "
            "direct, warm, never preachy. You celebrate good financial habits and flag risks honestly — "
            "but without judgement. You always know the *so what* behind the numbers.\n\n"

            "<core_mission>\n"
            "Your primary job is NOT to recite numbers — it is to turn raw transaction data into "
            "actionable financial intelligence. Every response must pass this test: "
            "\"Does this help the member make a better financial decision or understand something they didn't before?\"\n"
            "RESPONSE FORMAT — numbers → visual → insight → action:\n"
            "1. Lead with the single most useful number or direct answer (1-2 sentences max).\n"
            "2. Support it with a chart when the question involves trends, comparisons, or distributions.\n"
            "3. Provide a brief insight (what's surprising, what it means).\n"
            "4. End with ONE actionable next step or smart follow-up question.\n"
            "- Keep text under 80 words when possible — let charts carry the visual weight.\n"
            "- NEVER pad responses with filler phrases like 'Great question!' or 'As an AI...'.\n"
            "- NEVER fabricate, estimate, or hallucinate figures. Every number must come from the member's data.\n"
            "</core_mission>\n\n"

            "<response_rules_by_query_type>\n"
            "SPENDING / CATEGORIES / MERCHANTS:\n"
            "  → One-line direct answer with the key number, followed by a ranked top 3-5 list, "
            "a donut or bar chart, and a 1-sentence insight.\n"
            "TRENDS:\n"
            "  → Line chart first, then narrative with callouts for peaks/dips and overall direction.\n"
            "COMPARISON (Income vs Expenses):\n"
            "  → Net savings figure highlighted prominently, color-coded bar chart, month-by-month if useful.\n"
            "SAVINGS:\n"
            "  → Single headline metric ('You saved $X this month — up/down Y%'), savings rate %, brief suggestion.\n"
            "BUDGET:\n"
            "  → Prescriptive table: Category | 3-Mo Avg | Last Month | Suggested | Flag. "
            "Warning flags on overspent categories. End with one specific action to hit 20% savings.\n"
            "FULL FINANCIAL REPORT:\n"
            "  → Structured document: Summary card → Spending → Trends → Savings → Recommendations. "
            "Mix of charts + text, ending with 3 prioritized action items.\n"
            "</response_rules_by_query_type>\n\n"

            "<response_style>\n"
            "VOICE: Speak like a knowledgeable friend — direct, warm, never robotic or preachy.\n"
            "FORMAT:\n"
            "  - Short answers (1 clear stat): 2-3 sentences of prose. No bullets.\n"
            "  - Medium answers (comparing 2-5 items): Prose intro + tight bullet list.\n"
            "  - Long answers (deep analysis, full reports): Prose sections with bold labels.\n"
            "  - NEVER start a response with a heading, bold label, or markdown header.\n"
            "  - NEVER use emojis unless the member uses them first.\n"
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

            "<vola_product_awareness>\n"
            "When relevant to the member's situation, naturally weave in Vola features:\n"
            "- If spending exceeds income or balance is low: mention cash advance availability (up to $500, no interest)\n"
            "- If rent is a top category: mention rent reporting to credit bureaus to build credit\n"
            "- If many subscriptions detected: mention Vola's subscription tracker to find savings\n"
            "- If savings rate is poor: suggest setting up budget alerts in the Vola app\n"
            "- If member asks about credit: mention CreditMap dashboard and credit score tracking\n"
            "- If phone/mobile bills detected: mention Vola Mobile plans starting at $14.99/mo\n"
            "Do NOT force product mentions — only include them when genuinely helpful to the member's question.\n"
            "Keep product mentions to 1 sentence max, naturally integrated into your financial advice.\n"
            "</vola_product_awareness>\n\n"

            "<vola_membership_knowledge>\n"
            "Vola operates on a subscription membership model. Key facts you MUST know:\n"
            "PRICING: Vola charges a subscription fee — there are NO hidden fees, NO interest on cash advances, "
            "NO late fees, NO overdraft fees, NO credit check fees.\n"
            "WHAT MEMBERS GET:\n"
            "  - Cash Advances: $25–$500 deposited instantly (no interest, no credit check). "
            "Members start lower and increase limit with on-time repayments.\n"
            "  - Credit Building: Report rent/utility payments to credit bureaus. CreditMap dashboard "
            "shows credit score, credit mix, all loans & cards in one place.\n"
            "  - Expense Tracker: Automatic spend categorization, budget alerts, subscription detection.\n"
            "  - Subscription Tracker: Scans linked accounts, finds recurring charges (streaming, apps, gym, etc.), "
            "shows total monthly subscription outflow, flags unused subscriptions, allows cancel/pause from app, "
            "custom alerts (e.g. 'notify if a subscription exceeds $15').\n"
            "  - Vola Card: Mastercard debit via Evolve Bank & Trust — cashback, instant advance deposits.\n"
            "  - Vola Mobile: Wireless plans starting at $14.99/mo on AT&T 5G network. No contracts, eSIM support.\n"
            "  - Low Balance Alerts: Automatic warnings before you risk overdraft.\n"
            "ELIGIBILITY: Supports 6000+ banks and credit unions. Requires: account >3 months old, "
            "regular activity, average balance >$150, regular income/deposits.\n"
            "CANCELLATION: Cancel anytime. Refund available if features are incompatible with member's bank.\n"
            "When advising on subscriptions: analyze the member's actual recurring charges from their transaction data, "
            "calculate total monthly subscription cost, compare to income ratio, identify potential savings, "
            "and give specific actionable recommendations. Always answer in a structured, helpful format.\n"
            "</vola_membership_knowledge>\n\n"

            "<financial_intelligence>\n"
            "Apply these financial benchmarks when relevant — but only when supported by the data:\n"
            "  - 50/30/20 Rule: ~50% needs, ~30% wants, ~20% savings/debt.\n"
            "  - Savings target: 20% of gross income. Flag deviations.\n"
            "  - Emergency fund: 3-6 months of expenses is a healthy baseline.\n"
            "  - Spending spikes: A month 20%+ above average warrants a callout.\n"
            "  - Merchant concentration: If one merchant takes >15% of category spend, mention it.\n"
            "  - Positive reinforcement: Acknowledge improving trends and savings rate wins.\n"
            "PROACTIVE INSIGHTS: If you notice an anomaly or opportunity, briefly flag it with "
            "'One thing I noticed...' or 'Worth flagging:'.\n"
            "</financial_intelligence>\n\n"

            "<data_conventions>\n"
            "- Positive transaction_amount = EXPENSE (money going out)\n"
            "- Negative transaction_amount = INCOME (money coming in: salary, refunds, cashback, transfers)\n"
            "- Available data columns: user_id, user_name, transaction_date, "
            "transaction_amount, transaction_category_detail, merchant_name\n"
            "- All dates are in the member's local timezone. Refer to months by name (e.g. 'March') not number.\n"
            "</data_conventions>\n\n"

            "<visualization_tools>\n"
            "You have twelve charting tools. Use them proactively — a good chart explains in 2 seconds "
            "what takes 5 sentences. Rule: 1 chart per insight. Multiple charts = visual noise.\n"
            "  - plot_monthly_spending_trend(months, category_filter) → line chart of spending over time.\n"
            "  - plot_category_breakdown(period, top_n, months) → donut chart of category split.\n"
            "  - plot_income_vs_expense(months, show_net_line) → grouped bar chart of income vs spend.\n"
            "  - plot_top_merchants(top_n, months) → horizontal bar chart ranked by merchant spend.\n"
            "  - plot_weekly_pattern(months) → bar chart of avg daily spend by weekday.\n"
            "  - plot_category_trends(months, top_n) → stacked area chart of categories over time.\n"
            "  - plot_expense_distribution(bins) → histogram of transaction sizes.\n"
            "  - plot_cumulative_flow(months) → cumulative income vs expense area chart.\n"
            "  - plot_monthly_heatmap(months, top_n) → category × month spending heatmap.\n"
            "  - plot_savings_ratio(months) → monthly savings rate vs 20% target.\n"
            "  - plot_transaction_scatter(months) → transactions plotted by date and amount.\n"
            "  - plot_category_deepdive(category, months) → merchant breakdown within one category.\n"
            "Adjust 'months' to match the member's time reference: "
            "'last month' → 1, 'last quarter' → 3, 'last 6 months' → 6, 'last year' → 12.\n"
            "</visualization_tools>\n\n"

            "<follow_up_handling>\n"
            "When the member asks to EXPLAIN ('tell me more', 'elaborate', 'explain', 'why is that'):\n"
            "  - Do NOT call any chart tools. Write a rich narrative with context and benchmarks.\n"
            "When the member asks for a BETTER chart ('improve', 'different chart', 'redo', 'better graph'):\n"
            "  - Generate with meaningfully different parameters or chart type. Explain the change briefly.\n"
            "</follow_up_handling>\n\n"

            "<proactive_engagement>\n"
            "ALWAYS end your response with ONE smart, contextual follow-up question that:\n"
            "1. Is directly related to what you just discussed (not generic).\n"
            "2. Can be answered with a simple 'yes', 'no', or by picking a numbered option.\n"
            "3. Offers the member a useful next step or deeper insight they might want.\n\n"
            "FORMAT — use one of these patterns:\n"
            "  • Yes/No question: 'Would you like me to break down your grocery spending by merchant?'\n"
            "  • Numbered options: 'What would you like to explore next?\\n"
            "    1. See your spending trend over the last 6 months\\n"
            "    2. Compare your top categories month-by-month\\n"
            "    3. Get a detailed budget plan for next month'\n\n"
            "RULES:\n"
            "  - Make options SPECIFIC to the member's data and the current topic.\n"
            "  - Keep it to 2-3 options max for numbered choices.\n"
            "  - If the member responds with 'yes', 'no', '1', '2', or '3', treat it as answering your last question.\n"
            "  - NEVER repeat the same follow-up question you just asked.\n"
            "  - The follow-up should feel natural, not forced — like a friend continuing the conversation.\n"
            "</proactive_engagement>\n\n"

            f"<member_financial_profile>\n"
            f"Name: {profile['user_name']}\n"
            f"Analysis period: {profile['date_range']['start']} → {profile['date_range']['end']}\n"
            f"Total transactions analyzed: {profile['total_transactions']:,}\n"
            f"Average monthly spending: ${avg_monthly_spend:,.2f}\n"
            f"Average monthly income: ${avg_monthly_income:,.2f}\n"
            f"Financial position: Currently {savings_signal}\n"
            f"Top expense categories: {top_cats_display}\n"
            f"</member_financial_profile>\n"
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

    # ── Follow-up / proactive conversation resolution ──

    # Patterns that indicate a short follow-up reply rather than a new query
    _FOLLOWUP_PATTERNS = re.compile(
        r"^(yes|yeah|yep|yup|sure|absolutely|definitely|of course|go ahead|do it|"
        r"please|ok|okay|no|nah|nope|not really|"
        r"[1-9]|option\s*[1-9]|choice\s*[1-9]|"
        r"the\s+(first|second|third|1st|2nd|3rd)|both|all of (them|the above)|"
        r"sounds good|that works|let'?s do it|show me|tell me more|"
        r"i'?d (like|love|prefer|want)|why not)[\s!.,?]*$",
        re.IGNORECASE
    )

    def _resolve_followup(self, prompt: str, user_id: str) -> tuple[str, bool]:
        """If *prompt* is a short follow-up reply (yes/no/1/2/3 or a short
        topic word answering the AI's question), rewrite it into a full query
        by extracting the question from the last assistant turn in conversation
        history.  Returns (rewritten_prompt, is_followup).
        is_followup=True means the prompt should go directly to the LLM
        (skipping keyword intercepts) since it's a conversation continuation."""
        stripped = prompt.strip()

        # Only trigger for short inputs
        if len(stripped) > 60:
            return prompt, False

        conversation = self.cache.get_conversation(user_id)
        if not conversation:
            return prompt, False

        # Find the last assistant message
        last_assistant_msg = None
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break

        if not last_assistant_msg:
            return prompt, False

        # Extract the question the assistant asked (last sentence ending with ?)
        questions = re.findall(r'[^.!?\n]*\?', last_assistant_msg)
        if not questions:
            return prompt, False

        last_question = questions[-1].strip()

        # Extract numbered options from the assistant message (e.g., "1. ...", "2. ...")
        numbered_options = re.findall(r'^\s*(\d)[.)]\s*(.+)$', last_assistant_msg, re.MULTILINE)

        # Determine the user's intent from their short reply
        prompt_lower = stripped.lower().rstrip("!., ?")
        is_affirmative = prompt_lower in {
            "yes", "yeah", "yep", "yup", "sure", "absolutely", "definitely",
            "of course", "go ahead", "do it", "please", "ok", "okay",
            "sounds good", "that works", "why not",
        } or prompt_lower.startswith(("let's", "i'd like", "i'd love", "show me", "tell me"))

        is_negative = prompt_lower in {"no", "nah", "nope", "not really"}

        # Check for numbered selection (1, 2, 3, option 1, etc.)
        num_match = re.match(r"(?:option\s*|choice\s*)?(\d)", prompt_lower)
        ordinal_match = re.match(r"the\s+(first|second|third|1st|2nd|3rd)", prompt_lower)

        if is_affirmative:
            # User said yes → rewrite as the question itself as an instruction
            # Skip intercepts (is_followup=True) since keywords come from quoted context
            rewritten = f"Yes, please. Regarding your question: {last_question}"
            logger.info("Follow-up resolved: '%s' → '%s'", stripped, rewritten[:120])
            return rewritten, True
        elif is_negative:
            rewritten = (
                f"No thanks to: {last_question} — "
                f"Instead, what other insights can you give me about my finances?"
            )
            logger.info("Follow-up resolved: '%s' → '%s'", stripped, rewritten[:120])
            return rewritten, True
        elif num_match or ordinal_match:
            choice = num_match.group(1) if num_match else (
                {"first": "1", "second": "2", "third": "3",
                 "1st": "1", "2nd": "2", "3rd": "3"}[ordinal_match.group(1)]
            )
            # Try to find the exact text for the chosen option
            chosen_text = None
            for opt_num, opt_text in numbered_options:
                if opt_num == choice:
                    # Strip markdown bold markers for a cleaner prompt
                    chosen_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', opt_text.strip())
                    break
            if chosen_text:
                # The option text IS the intended query — route through normal intercepts
                # (is_followup=False so subscription/budget/etc handlers can catch it)
                logger.info("Follow-up option %s resolved: '%s' → '%s'", choice, stripped, chosen_text[:120])
                return chosen_text, False
            else:
                # No numbered options found — include the full options block for context
                opts_block = "\n".join(f"  {n}. {t}" for n, t in numbered_options) if numbered_options else ""
                rewritten = f"I pick option {choice} from: {last_question}" + (f"\nOptions were:\n{opts_block}" if opts_block else "")
                logger.info("Follow-up resolved: '%s' → '%s'", stripped, rewritten[:120])
                return rewritten, True
        elif len(stripped.split()) <= 5:
            # Short topic reply (e.g. "groceries", "pet supplies", "flights")
            # answering a question the assistant just asked.
            # Keep the rewrite minimal — just the topic name so the
            # category matcher picks it up without triggering other
            # intent handlers (avoid words like "category", "breakdown",
            # "detail", "trend", "spending" which match generic handlers).
            rewritten = f"show {stripped} transactions"
            logger.info("Follow-up topic resolved: '%s' → '%s'", stripped, rewritten[:120])
            return rewritten, False
        else:
            # Longer reply that's still within 60 chars — pass along with context
            if self._FOLLOWUP_PATTERNS.match(stripped):
                rewritten = f"Regarding your question '{last_question}', my answer is: {stripped}"
                logger.info("Follow-up resolved: '%s' → '%s'", stripped, rewritten[:120])
                return rewritten, True
            return prompt, False

    def _execute_tool_calls(self, tool_calls: list, user_id: str, prompt: str = "") -> list[dict]:
        paths: list = []
        # Detect time period from the user's prompt to override chart params
        max_date = self.df[self.df["user_id"] == user_id]["transaction_date"].max()
        detected_period, _ = self._extract_time_period(prompt, max_date) if prompt else (None, "")
        detected_months = None
        if isinstance(detected_period, int):
            detected_months = detected_period
        elif isinstance(detected_period, tuple):
            detected_months = 1

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

            # Override chart period with detected time period from user's query
            if detected_months is not None:
                if fn_name == "plot_category_breakdown":
                    args["months"] = detected_months
                elif "months" in args or fn_name in (
                    "plot_monthly_spending_trend", "plot_top_merchants",
                    "plot_income_vs_expense", "plot_category_trends",
                ):
                    args["months"] = detected_months

            try:
                path = VIZ_FUNCTIONS[fn_name](df=self.df, **args)
                if path:
                    paths.append({"path": path, "fn": fn_name, "kwargs": {k: v for k, v in args.items() if k != "user_id"}})
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
        """Strip spurious tool-call artefacts, raw JSON blocks, and markdown noise
        that the LLM sometimes injects into its prose output."""
        # Remove <tool_call> XML (closed and unclosed)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL)

        # Remove markdown fenced code blocks containing JSON tool-call payloads
        # e.g.  ```json\n{"user_id": ...}\n```
        text = re.sub(r"```(?:json)?\s*\n?\s*\{[^}]*\"(?:user_id|months|top_n|period|category|show_net_line|bins)\"[^`]*```", "", text, flags=re.DOTALL)

        # Remove inline bare JSON objects that look like tool call args
        # e.g.  {"user_id": "ValentinaCruz", "months": 30, ...}
        text = re.sub(r'\{\s*"user_id"\s*:.*?\}', '', text, flags=re.DOTALL)

        # Remove orphan ```json or ``` markers left behind
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Remove *Chart:* labels that precede tool calls (now removed)
        text = re.sub(r"\*Chart:?\*[^\n]*\n?", "", text)

        # Collapse 3+ consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace raw DB category names (e.g. RENT_HOUSING) with friendly labels
        for raw, friendly in _CATEGORY_DISPLAY.items():
            text = text.replace(raw, friendly)
        return text.strip()

    # ── Time-period extraction ──────────────────────────────────

    @staticmethod
    def _extract_time_period(prompt: str, max_date: pd.Timestamp) -> tuple[int | None, str]:
        """Parse a time-period reference from the user's prompt.

        Returns (months: int | None, label: str).
        months=None means no period detected (use all data).
        label is a human-friendly string like "last month" or "last 3 months".
        """
        p = prompt.lower()

        # "this month" / "current month"
        if re.search(r"\b(this|current)\s+month\b", p):
            return 1, "this month"

        # "last N months" / "past N months" / "previous N months"
        m = re.search(r"\b(?:last|past|previous|recent)\s+(\d+)\s+months?\b", p)
        if m:
            n = int(m.group(1))
            return n, f"last {n} month{'s' if n > 1 else ''}"

        # "last month" / "previous month" / "past month" (exactly 1 month)
        if re.search(r"\b(?:last|past|previous)\s+month\b", p):
            return 1, "last month"

        # "last N weeks"
        m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+weeks?\b", p)
        if m:
            weeks = int(m.group(1))
            # Convert weeks to approximate months (min 1)
            months = max(1, round(weeks / 4.3))
            return months, f"last {weeks} week{'s' if weeks > 1 else ''}"

        # "last week"
        if re.search(r"\b(?:last|past|previous)\s+week\b", p):
            return 1, "last week"

        # "last year" / "past year"
        if re.search(r"\b(?:last|past|previous)\s+year\b", p):
            return 12, "last year"

        # "last 6 months", "last quarter", etc.
        if re.search(r"\b(?:last|past|previous)\s+quarter\b", p):
            return 3, "last quarter"

        # "last half year" / "last 6 months"
        if re.search(r"\b(?:last|past)\s+half\s+year\b", p):
            return 6, "last 6 months"

        # "in January", "in March 2025", etc.
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
            "oct": 10, "nov": 11, "dec": 12,
        }
        m = re.search(r"\bin\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{4})?\b", p)
        if m:
            month_num = month_names.get(m.group(1), 0)
            year = int(m.group(2)) if m.group(2) else max_date.year
            return (month_num, year), f"{m.group(1).capitalize()}{' ' + str(year) if m.group(2) else ''}"

        return None, "all time"

    @staticmethod
    def _apply_time_filter(df: pd.DataFrame, period, max_date: pd.Timestamp) -> pd.DataFrame:
        """Filter a DataFrame by the period returned from _extract_time_period.

        period can be:
          - None → no filter (all data)
          - int  → last N months from max_date
          - tuple (month_num, year) → specific calendar month
        """
        if period is None:
            return df
        if isinstance(period, tuple):
            month_num, year = period
            return df[
                (df["transaction_date"].dt.month == month_num) &
                (df["transaction_date"].dt.year == year)
            ]
        # int → last N months
        cutoff = max_date - pd.DateOffset(months=period)
        return df[df["transaction_date"] >= cutoff]

    def _summarise_from_data(self, user_df: pd.DataFrame, prompt: str, chart_paths: list[str]) -> str:
        """Generate a narrative answer by assembling the relevant data slice
        and sending it to the LLM so the response is always natural, specific
        to the user's question, and grounded in real numbers."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]
        prompt_lower = prompt.lower()

        # ── Dynamic time-period filtering ──
        max_date = user_df["transaction_date"].max()
        period, period_label = self._extract_time_period(prompt, max_date)
        if period is not None:
            expenses = self._apply_time_filter(expenses, period, max_date)
            income = self._apply_time_filter(income, period, max_date)

        total_exp = round(expenses["transaction_amount"].sum(), 2)
        total_inc = round(abs(income["transaction_amount"].sum()), 2)
        net = total_inc - total_exp
        savings_rate = (net / total_inc * 100) if total_inc > 0 else 0

        period_qualifier = f" ({period_label})" if period is not None else ""

        # ── Build targeted data context based on what the user asked ──
        data_lines: list[str] = [
            f"Period: {period_label if period is not None else 'all time'}",
            f"Total spent: ${total_exp:,.2f} | Total earned: ${total_inc:,.2f}",
            f"Net: ${net:+,.2f} | Savings rate: {savings_rate:.1f}%",
        ]

        # Detect specific category in the prompt
        _categories = expenses["transaction_category_detail"].str.lower().unique().tolist()
        _matched_cat = None
        _cat_stop = {
            "income", "expense", "expenses", "save", "saving", "savings",
            "compare", "total", "amount", "budget", "money", "trend",
            "monthly", "daily", "weekly", "average", "payment",
            "subscription", "subscriptions", "recurring", "streaming",
            "cancel", "membership",
        }
        for _c in _categories:
            _parts = _c.split("_")
            if _c in prompt_lower:
                _matched_cat = _c
                break
            for _p in _parts:
                if len(_p) > 3 and _p in prompt_lower and _p not in _cat_stop:
                    _matched_cat = _c
                    break
            if _matched_cat:
                break

        # Always include top categories
        top_cats = (
            expenses.groupby("transaction_category_detail")["transaction_amount"]
            .sum().nlargest(5)
        )
        data_lines.append("Top 5 categories: " + ", ".join(
            f"{friendly_category(c)}: ${a:,.2f} ({a/total_exp*100:.1f}%)" if total_exp else f"{friendly_category(c)}: ${a:,.2f}"
            for c, a in top_cats.items()
        ))

        # Add merchant data when relevant
        if _matched_cat:
            cat_upper = _matched_cat.upper()
            cat_txns = expenses[expenses["transaction_category_detail"].str.upper() == cat_upper]
            cat_label = friendly_category(cat_upper)
            if not cat_txns.empty:
                cat_total = cat_txns["transaction_amount"].sum()
                data_lines.append(f"\n--- {cat_label} category detail ---")
                data_lines.append(f"Total: ${cat_total:,.2f} ({cat_total/total_exp*100:.1f}% of all spending) across {len(cat_txns)} transactions")
                data_lines.append(f"Average per transaction: ${cat_txns['transaction_amount'].mean():,.2f}")
                cat_monthly = cat_txns.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
                if not cat_monthly.empty and len(cat_monthly) >= 2:
                    data_lines.append(f"Monthly avg: ${cat_monthly.mean():,.2f} | Peak: {cat_monthly.idxmax().strftime('%b %Y')} (${cat_monthly.max():,.2f})")
                if "merchant_name" in cat_txns.columns:
                    top_m = cat_txns.groupby("merchant_name")["transaction_amount"].sum().nlargest(5)
                    data_lines.append("Merchants in this category: " + ", ".join(
                        f"{m}: ${a:,.2f} ({a/cat_total*100:.1f}%)" for m, a in top_m.items()
                    ))

        if any(w in prompt_lower for w in ["merchant", "store", "shop", "vendor", "places", "where do i"]):
            top_merchants = expenses.groupby("merchant_name")["transaction_amount"].agg(["sum", "count"]).nlargest(10, "sum")
            data_lines.append("\n--- Top merchants ---")
            for name, row in top_merchants.iterrows():
                pct = (row["sum"] / total_exp * 100) if total_exp else 0
                cat = expenses[expenses["merchant_name"] == name]["transaction_category_detail"].mode()
                cat_str = f" [{friendly_category(cat.iloc[0])}]" if not cat.empty else ""
                data_lines.append(f"{name}: ${row['sum']:,.2f} ({pct:.1f}%) — {int(row['count'])} txns{cat_str}")

        if any(w in prompt_lower for w in ["trend", "over time", "month"]):
            monthly_exp = expenses.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
            if not monthly_exp.empty:
                data_lines.append("\n--- Monthly spending ---")
                for dt, amt in monthly_exp.items():
                    data_lines.append(f"{dt.strftime('%b %Y')}: ${amt:,.2f}")

        if any(w in prompt_lower for w in ["most", "top", "highest", "breakdown", "category", "where"]) and not _matched_cat:
            # Already included top_cats above — add merchant context too
            top_merchants = expenses.groupby("merchant_name")["transaction_amount"].sum().nlargest(5)
            data_lines.append("Top 5 merchants: " + ", ".join(
                f"{m}: ${a:,.2f}" for m, a in top_merchants.items()
            ))

        if any(w in prompt_lower for w in ["saving", "save", "income", "expense", "financial", "net"]):
            monthly_exp = expenses.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
            monthly_inc = abs(income.set_index("transaction_date").resample("ME")["transaction_amount"].sum())
            if not monthly_exp.empty and len(monthly_exp) >= 2:
                data_lines.append(f"\nRecent months spending: " + ", ".join(
                    f"{d.strftime('%b %Y')}: ${a:,.2f}" for d, a in monthly_exp.tail(3).items()
                ))

        data_context = "\n".join(data_lines)

        # ── Send to LLM for a natural, query-specific response ──
        try:
            llm_result = self.llm.chat(messages=[
                {"role": "system", "content": (
                    "You are Vola AI, a smart financial assistant. The member asked a question "
                    "and their actual financial data is below. Give a helpful, specific response "
                    "that DIRECTLY answers their exact question using the data provided. "
                    "Be precise with numbers from the data. Keep it concise (under 120 words). "
                    "Use markdown (bold for key figures, bullets for lists). "
                    "Do NOT repeat data the user didn't ask about. "
                    "Do NOT start with a heading. Speak like a knowledgeable friend."
                )},
                {"role": "user", "content": f"Member's question: {prompt}\n\n{data_context}"},
            ])
            if not llm_result.get("error") and llm_result.get("content", "").strip():
                response = self._clean_response(llm_result["content"].strip())
                if chart_paths:
                    n = len(chart_paths)
                    response += f"\nI've generated {n} chart{'s' if n > 1 else ''} below to visualize this for you."
                return response
        except Exception:
            pass

        # ── Template fallback only if LLM call itself fails ──
        lines: list[str] = []
        if _matched_cat:
            cat_upper = _matched_cat.upper()
            cat_txns = expenses[expenses["transaction_category_detail"].str.upper() == cat_upper]
            cat_label = friendly_category(cat_upper)
            if not cat_txns.empty:
                cat_total = cat_txns["transaction_amount"].sum()
                pct = (cat_total / total_exp * 100) if total_exp else 0
                lines.append(
                    f"You've spent **${cat_total:,.2f}** on **{cat_label}**{period_qualifier} across "
                    f"{len(cat_txns)} transactions — that's {pct:.1f}% of your total spending."
                )
                if "merchant_name" in cat_txns.columns:
                    top_merch = cat_txns.groupby("merchant_name")["transaction_amount"].sum().nlargest(3)
                    if not top_merch.empty:
                        lines.append(f"\nTop merchants in {cat_label}:")
                        for merch, amt in top_merch.items():
                            m_pct = (amt / cat_total * 100) if cat_total else 0
                            lines.append(f"  • {merch}: ${amt:,.2f} ({m_pct:.1f}%)")
            else:
                lines.append(f"No transactions found in the {cat_label} category.")
        elif any(w in prompt_lower for w in ["merchant", "store", "shop", "vendor", "places"]):
            top_m = expenses.groupby("merchant_name")["transaction_amount"].agg(["sum", "count"]).nlargest(5, "sum")
            if not top_m.empty:
                lines.append(f"Here are your top merchants{period_qualifier}:")
                for rank, (name, row) in enumerate(top_m.iterrows(), 1):
                    pct = (row["sum"] / total_exp * 100) if total_exp else 0
                    lines.append(f"  {rank}. **{name}**: ${row['sum']:,.2f} ({pct:.1f}%) — {int(row['count'])} transactions")
        else:
            lines.append(
                f"Here's the quick picture: {len(user_df):,} total transactions, "
                f"${total_exp:,.2f} spent and ${total_inc:,.2f} earned — "
                f"{'saving' if net >= 0 else 'deficit of'} ${abs(net):,.2f}."
            )
            if not top_cats.empty:
                lines.append(f"Top category: **{friendly_category(top_cats.index[0])}** at ${top_cats.iloc[0]:,.2f}.")

        if chart_paths:
            n = len(chart_paths)
            lines.append(f"\nI've generated {n} chart{'s' if n > 1 else ''} below to visualize this for you.")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    #  _is_full_report_intent()  —  Detect "full report" requests
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _is_full_report_intent(prompt: str) -> bool:
        p = prompt.lower().strip()
        # Exact phrases
        if any(phrase in p for phrase in [
            "full report", "full financial report", "financial report",
            "complete report", "comprehensive report", "detailed report",
            "annual report", "give me everything", "show me everything",
            "full analysis", "complete analysis", "full dashboard",
            "complete dashboard", "full summary", "complete summary",
        ]):
            return True
        # "give me a full ___" patterns
        if re.search(r"\b(give|show|generate|create|run|build|produce)\b.*\b(full|complete|comprehensive|detailed|entire)\b", p):
            return True
        return False

    # ══════════════════════════════════════════════════════════════
    #  _comprehensive_report()  —  Multi-section financial report
    # ══════════════════════════════════════════════════════════════

    def _comprehensive_report(self, user_df: pd.DataFrame, user_id: str) -> tuple[str, list]:
        """Generate a comprehensive multi-section financial report with all 12 chart types.
        Returns (report_text, chart_paths)."""
        expenses = user_df[user_df["transaction_amount"] > 0]
        income = user_df[user_df["transaction_amount"] < 0]
        user_name = user_df["user_name"].iloc[0]

        total_exp = round(expenses["transaction_amount"].sum(), 2)
        total_inc = round(abs(income["transaction_amount"].sum()), 2)
        net_savings = round(total_inc - total_exp, 2)

        monthly_exp = expenses.set_index("transaction_date").resample("ME")["transaction_amount"].sum()
        monthly_inc = income.set_index("transaction_date").resample("ME")["transaction_amount"].sum().abs()
        num_months = max(len(monthly_exp), 1)
        savings_rate = (net_savings / total_inc * 100) if total_inc else 0

        top_cats = expenses.groupby("transaction_category_detail")["transaction_amount"].sum().nlargest(10)
        top_merchants = expenses.groupby("merchant_name")["transaction_amount"].sum().nlargest(10)

        avg_monthly_inc = total_inc / num_months if num_months else 0
        avg_monthly_exp = total_exp / num_months if num_months else 0
        avg_monthly_sav = net_savings / num_months if num_months else 0

        chart_paths: list = []

        def _try_chart(fn_name: str, **kwargs):
            try:
                p = VIZ_FUNCTIONS[fn_name](df=self.df, user_id=user_id, **kwargs)
                if p:
                    chart_paths.append({"path": p, "fn": fn_name, "kwargs": kwargs})
            except Exception:
                pass

        lines: list[str] = []

        # ────────────────────────────────────────────
        # Section 1: Executive Summary
        # ────────────────────────────────────────────
        lines.append(f"# Full Financial Report — {user_name}\n")
        lines.append(f"**Report Period:** {num_months} months | "
                     f"**Transactions:** {len(user_df):,} | "
                     f"**Generated:** {pd.Timestamp.now().strftime('%B %d, %Y')}\n")

        lines.append("## 1. Executive Summary\n")
        if net_savings >= 0:
            verdict = f"You're in a **strong financial position** — earning more than you spend with a **{savings_rate:.1f}% savings rate**."
        elif savings_rate > -5:
            verdict = f"You're **slightly over-budget** with a **{savings_rate:.1f}% deficit**. Small adjustments can fix this."
        else:
            verdict = f"You're running a **{abs(savings_rate):.1f}% deficit** — spending significantly exceeds income. Immediate action recommended."
        lines.append(verdict + "\n")

        lines.append("| Metric | Amount |")
        lines.append("|--------|--------|")
        lines.append(f"| Total Income | **${total_inc:,.2f}** |")
        lines.append(f"| Total Expenses | **${total_exp:,.2f}** |")
        lines.append(f"| Net {'Savings' if net_savings >= 0 else 'Deficit'} | **${abs(net_savings):,.2f}** |")
        lines.append(f"| Savings Rate | **{savings_rate:.1f}%** (target: 20%) |")
        lines.append(f"| Monthly Income (avg) | **${avg_monthly_inc:,.2f}** |")
        lines.append(f"| Monthly Expenses (avg) | **${avg_monthly_exp:,.2f}** |")
        lines.append(f"| Monthly Savings (avg) | **${avg_monthly_sav:,.2f}** |")
        lines.append("")

        # ────────────────────────────────────────────
        # Section 2: Income vs Expenses
        # ────────────────────────────────────────────
        lines.append("## 2. Income vs Expenses\n")
        if not monthly_inc.empty and not monthly_exp.empty:
            surplus_months = sum(1 for i, e in zip(monthly_inc, monthly_exp) if i > e)
            deficit_months = num_months - surplus_months
            lines.append(f"Over {num_months} months, you had **{surplus_months} surplus months** "
                        f"and **{deficit_months} deficit months**.\n")
            if len(monthly_exp) >= 2:
                last_exp = monthly_exp.iloc[-1]
                prev_exp = monthly_exp.iloc[-2]
                change = last_exp - prev_exp
                direction = "increased" if change > 0 else "decreased"
                lines.append(f"Your most recent month's spending **{direction}** by **${abs(change):,.2f}** "
                            f"compared to the prior month.\n")

            best_month = monthly_inc.idxmax().strftime("%b %Y") if not monthly_inc.empty else "N/A"
            worst_month = monthly_exp.idxmax().strftime("%b %Y") if not monthly_exp.empty else "N/A"
            lines.append(f"  - Best income month: **{best_month}** (${monthly_inc.max():,.2f})")
            lines.append(f"  - Highest spending month: **{worst_month}** (${monthly_exp.max():,.2f})")
            lines.append(f"  - Lowest spending month: **{monthly_exp.idxmin().strftime('%b %Y')}** (${monthly_exp.min():,.2f})")
            lines.append("")

        _try_chart("plot_income_vs_expense", months=num_months)
        _try_chart("plot_savings_ratio", months=num_months)

        # ────────────────────────────────────────────
        # Section 3: Spending Analysis
        # ────────────────────────────────────────────
        lines.append("## 3. Spending Breakdown\n")
        lines.append("**Top Spending Categories:**\n")
        for i, (cat, amt) in enumerate(top_cats.items(), 1):
            pct = (amt / total_exp * 100) if total_exp else 0
            lines.append(f"  {i}. **{friendly_category(cat)}** — ${amt:,.2f} ({pct:.1f}%)")
        lines.append("")

        # Concentration analysis
        if len(top_cats) >= 3:
            top3_pct = (top_cats.iloc[:3].sum() / total_exp * 100) if total_exp else 0
            lines.append(f"Your top 3 categories account for **{top3_pct:.0f}%** of total spending. "
                        f"{'That\'s highly concentrated — diversifying could reduce risk.' if top3_pct > 70 else 'A reasonably balanced distribution.'}\n")

        _try_chart("plot_category_breakdown", period="all_time")
        _try_chart("plot_category_trends", months=num_months)

        # ────────────────────────────────────────────
        # Section 4: Merchant Analysis
        # ────────────────────────────────────────────
        lines.append("## 4. Merchant Analysis\n")
        lines.append("**Top Merchants by Spend:**\n")
        for i, (merch, amt) in enumerate(top_merchants.items(), 1):
            pct = (amt / total_exp * 100) if total_exp else 0
            visit_count = len(expenses[expenses["merchant_name"] == merch])
            avg_per_visit = amt / visit_count if visit_count else 0
            lines.append(f"  {i}. **{merch}** — ${amt:,.2f} ({pct:.1f}%) | "
                        f"{visit_count} visits, avg ${avg_per_visit:,.2f}/visit")
        lines.append("")

        _try_chart("plot_top_merchants", months=num_months)
        # Deep-dive into the biggest category
        if not top_cats.empty:
            _try_chart("plot_category_deepdive", category=top_cats.index[0], months=num_months)

        # ────────────────────────────────────────────
        # Section 5: Cash Flow & Trends
        # ────────────────────────────────────────────
        lines.append("## 5. Cash Flow & Trends\n")
        if not monthly_exp.empty:
            # Trend direction
            if len(monthly_exp) >= 3:
                recent_3 = monthly_exp.iloc[-3:].mean()
                older_3 = monthly_exp.iloc[-6:-3].mean() if len(monthly_exp) >= 6 else monthly_exp.iloc[:3].mean()
                if recent_3 > older_3 * 1.1:
                    trend_text = "Your spending is **trending upward** — recent months are higher than earlier ones."
                elif recent_3 < older_3 * 0.9:
                    trend_text = "Your spending is **trending downward** — good discipline showing in recent months."
                else:
                    trend_text = "Your spending has been **relatively stable** over time."
                lines.append(trend_text + "\n")

            high_month = monthly_exp.idxmax().strftime("%B %Y")
            low_month = monthly_exp.idxmin().strftime("%B %Y")
            volatility = monthly_exp.std() / monthly_exp.mean() * 100 if monthly_exp.mean() else 0
            lines.append(f"  - Spending volatility: **{volatility:.0f}%** "
                        f"{'(high — your spending swings significantly month to month)' if volatility > 30 else '(moderate)' if volatility > 15 else '(low — very consistent spending)'}")
            lines.append(f"  - Peak: **{high_month}** (${monthly_exp.max():,.2f})")
            lines.append(f"  - Trough: **{low_month}** (${monthly_exp.min():,.2f})")
            lines.append("")

        _try_chart("plot_monthly_spending_trend", months=num_months)
        _try_chart("plot_cumulative_flow", months=num_months)

        # ────────────────────────────────────────────
        # Section 6: Behavioral Patterns
        # ────────────────────────────────────────────
        lines.append("## 6. Behavioral Patterns\n")

        # Weekly pattern
        exp_copy = expenses.copy()
        exp_copy["dow"] = exp_copy["transaction_date"].dt.day_name()
        day_avg = exp_copy.groupby("dow")["transaction_amount"].mean()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_avg = day_avg.reindex(day_order).fillna(0)
        if not day_avg.empty:
            peak_day = day_avg.idxmax()
            low_day = day_avg.idxmin()
            weekend_avg = day_avg[["Saturday", "Sunday"]].mean()
            weekday_avg = day_avg[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean()
            lines.append(f"  - **Highest spending day:** {peak_day} (avg ${day_avg.max():,.2f})")
            lines.append(f"  - **Lowest spending day:** {low_day} (avg ${day_avg.min():,.2f})")
            lines.append(f"  - **Weekday avg:** ${weekday_avg:,.2f} vs **Weekend avg:** ${weekend_avg:,.2f}")
            lines.append("")

        # Transaction size distribution
        median_txn = expenses["transaction_amount"].median()
        mean_txn = expenses["transaction_amount"].mean()
        large_txns = expenses[expenses["transaction_amount"] > mean_txn * 2]
        lines.append(f"  - **Median transaction:** ${median_txn:,.2f} | **Mean:** ${mean_txn:,.2f}")
        lines.append(f"  - **Large transactions (>2x mean):** {len(large_txns)} "
                    f"(${large_txns['transaction_amount'].sum():,.2f} total)")
        lines.append("")

        _try_chart("plot_weekly_pattern", months=num_months)
        _try_chart("plot_expense_distribution")
        _try_chart("plot_transaction_scatter", months=min(num_months, 6))
        _try_chart("plot_monthly_heatmap", months=min(num_months, 12))

        # ────────────────────────────────────────────
        # Section 7: Recommendations
        # ────────────────────────────────────────────
        lines.append("## 7. Key Takeaways & Recommendations\n")
        recs = []
        if savings_rate >= 30:
            recs.append("Your savings rate is **excellent** — consider investing surplus funds for long-term growth.")
        elif savings_rate >= 20:
            recs.append("You're meeting the 20% savings target — maintain this discipline and consider automating savings.")
        elif savings_rate > 0:
            gap = 20 - savings_rate
            monthly_gap = avg_monthly_inc * gap / 100
            recs.append(f"You're saving {savings_rate:.1f}% — **${monthly_gap:,.0f}/month more** would hit the 20% target.")
        else:
            recs.append(f"**Priority: close the deficit.** You're spending ${abs(net_savings / num_months):,.0f}/month more than you earn.")

        if not top_cats.empty:
            top_cat_name = friendly_category(top_cats.index[0])
            top_cat_pct = (top_cats.iloc[0] / total_exp * 100) if total_exp else 0
            if top_cat_pct > 40:
                recs.append(f"**{top_cat_name}** dominates at {top_cat_pct:.0f}% of spending — even a 10% reduction would save ${top_cats.iloc[0] * 0.1:,.0f}.")

        if len(large_txns) > 0:
            recs.append(f"You had **{len(large_txns)} large transactions** totalling ${large_txns['transaction_amount'].sum():,.2f} — review these for potential savings.")

        if volatility > 30:
            recs.append("Spending volatility is high — setting monthly budgets per category could help smooth out the swings.")

        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        # Footer
        n = len(chart_paths)
        lines.append(f"---\n*This report includes **{n} visualizations** below covering all aspects of your finances.*")

        return "\n".join(lines), chart_paths

    def _fallback_response(self, user_df: pd.DataFrame, prompt: str, user_id: str,
                            conversation: list = None) -> tuple[str, list[str]]:
        """Generate a data-driven, question-aware answer + charts when the LLM is unreachable.
        Handles diverse questions including follow-ups, merchant queries, specific categories,
        graph requests, and more. Returns (response_text, chart_paths)."""
        all_expenses = user_df[user_df["transaction_amount"] > 0]
        all_income = user_df[user_df["transaction_amount"] < 0]
        user_name = user_df["user_name"].iloc[0]
        prompt_lower = prompt.lower()
        lines: list[str] = []
        chart_paths: list[str] = []

        # ── Dynamic time-period filtering ──
        max_date = user_df["transaction_date"].max()
        period, period_label = self._extract_time_period(prompt, max_date)
        if period is not None:
            expenses = self._apply_time_filter(all_expenses, period, max_date)
            income = self._apply_time_filter(all_income, period, max_date)
        else:
            expenses = all_expenses
            income = all_income
        # Convert period to months int for chart params
        if isinstance(period, tuple):
            period_months = 1
        elif isinstance(period, int):
            period_months = period
        else:
            period_months = None  # all time

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

        # Chart period: use detected period or default
        _chart_months = period_months if period_months else None
        _chart_period = "last_month" if period_months == 1 else f"last_{period_months}_months" if period_months else None

        # Get viz state for follow-up context
        viz_state = self.cache.get_viz_state(user_id)
        last_chart = viz_state.get("last_chart_type", "") if viz_state else ""

        # ── Helper to safely generate charts ──
        def _try_chart(fn_name: str, **kwargs):
            try:
                p = VIZ_FUNCTIONS[fn_name](df=self.df, user_id=user_id, **kwargs)
                if p:
                    chart_paths.append({"path": p, "fn": fn_name, "kwargs": kwargs})
                    # Update viz_state so follow-up "better graph" knows what was last shown
                    self.cache.set_viz_state(
                        user_id,
                        chart_type=fn_name,
                        axes=kwargs,
                        filters={k: v for k, v in kwargs.items() if k != "user_id"},
                    )
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
            "subscription", "subscriptions", "recurring", "streaming",
            "cancel", "membership",
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
            last_filters = viz_state.get("filters", {}) if viz_state else {}
            if last_chart == "plot_category_breakdown":
                lines.append("Here's a deeper look — your spending broken down by category over time, plus your top merchants:")
                # Show category trends (stacked area) + top merchants bar — genuinely different from the donut
                _try_chart("plot_category_trends", months=last_filters.get("months", 6))
                _try_chart("plot_top_merchants", months=last_filters.get("months", 6))
                _try_chart("plot_expense_distribution")
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

        elif (any(w in prompt_lower for w in ["how should", "advice", "suggest", "recommend", "plan", "budget", "next month",
                                              "what should", "can i afford", "afford", "reduce", "cut"])
              and not any(w in prompt_lower for w in ["subscription", "subscriptions", "recurring", "streaming"])):
            # ── Prescriptive budget recommendation ──
            # Detect multi-month requests: "next 3 months", "next quarter", "6 months", etc.
            plan_months = 1  # default: single-month budget
            _mo_match = re.search(r"(?:next|coming|upcoming|for)\s+(\d+)\s*(?:months?|mos?)", prompt_lower)
            if _mo_match:
                plan_months = min(int(_mo_match.group(1)), 12)
            elif any(w in prompt_lower for w in ["quarter", "3 month", "three month"]):
                plan_months = 3
            elif any(w in prompt_lower for w in ["half year", "6 month", "six month"]):
                plan_months = 6

            # Use last 3 months as baseline for category averages
            three_mo_cutoff = max_date - pd.DateOffset(months=3)
            last_mo_cutoff = max_date - pd.DateOffset(months=1)
            recent_exp = all_expenses[all_expenses["transaction_date"] > three_mo_cutoff]
            last_mo_exp = all_expenses[(all_expenses["transaction_date"] > last_mo_cutoff)]

            avg_monthly_total = round(recent_exp["transaction_amount"].sum() / 3, 2) if not recent_exp.empty else 0
            last_mo_total = round(last_mo_exp["transaction_amount"].sum(), 2) if not last_mo_exp.empty else 0

            # Category averages (3-month) vs last month actuals
            cat_3mo_avg = recent_exp.groupby("transaction_category_detail")["transaction_amount"].sum() / 3
            cat_last_mo = last_mo_exp.groupby("transaction_category_detail")["transaction_amount"].sum()

            # Monthly income
            recent_inc = abs(all_income[all_income["transaction_date"] > three_mo_cutoff]["transaction_amount"].sum()) / 3
            current_savings = round(recent_inc - avg_monthly_total, 2)
            savings_pct = round((current_savings / recent_inc * 100), 1) if recent_inc > 0 else 0
            target_20_spend = round(recent_inc * 0.80, 2)

            # Build top categories table
            all_cats_3mo = cat_3mo_avg.sort_values(ascending=False)
            top_budget_cats = all_cats_3mo.head(8)

            # Discretionary categories where we can suggest cuts
            _essential = {"rent", "housing", "mortgage", "insurance", "utilities", "internet", "loan", "emi"}

            # Headline
            if current_savings >= 0:
                lines.append(
                    f"**${avg_monthly_total:,.0f}/mo** avg spend | **${recent_inc:,.0f}/mo** income | "
                    f"**{savings_pct}%** savings rate"
                )
            else:
                lines.append(
                    f"**${avg_monthly_total:,.0f}/mo** avg spend | **${recent_inc:,.0f}/mo** income | "
                    f"**-${abs(current_savings):,.0f}/mo** deficit"
                )
            lines.append("")

            if plan_months <= 1:
                # ── Single-month budget (original behaviour) ──
                lines.append("### Recommended Budget for Next Month")
                lines.append("")
                lines.append("| Category | 3-Mo Avg | Last Month | Suggested | Flag |")
                lines.append("|----------|----------|------------|-----------|------|")

                for cat, avg_amt in top_budget_cats.items():
                    cat_label = friendly_category(cat)
                    avg_val = round(avg_amt, 0)
                    last_val = round(cat_last_mo.get(cat, 0), 0)

                    is_essential = any(e in cat.lower() for e in _essential)
                    overspent = last_val > avg_val * 1.15 if avg_val > 0 else False
                    underspent = last_val < avg_val * 0.80 if avg_val > 0 else False

                    if overspent and not is_essential:
                        suggested = round(avg_val * 0.90, 0)
                        flag = "⚠️ Over"
                    elif overspent and is_essential:
                        suggested = avg_val
                        flag = "⚠️ Over"
                    elif underspent:
                        suggested = round(last_val, 0)
                        flag = "✅ Good"
                    else:
                        suggested = avg_val
                        flag = "→ On track"

                    lines.append(
                        f"| {cat_label} | ${avg_val:,.0f} | ${last_val:,.0f} | **${suggested:,.0f}** | {flag} |"
                    )

                others_avg = round(all_cats_3mo.iloc[8:].sum(), 0) if len(all_cats_3mo) > 8 else 0
                others_last = round(sum(cat_last_mo.get(c, 0) for c in all_cats_3mo.index[8:]), 0) if len(all_cats_3mo) > 8 else 0
                if others_avg > 0 or others_last > 0:
                    lines.append(f"| Other | ${others_avg:,.0f} | ${others_last:,.0f} | **${others_avg:,.0f}** | |")
            else:
                # ── Multi-month budget plan ──
                # Generate future month names starting from the month after max_date
                future_months = []
                for mi in range(1, plan_months + 1):
                    fm = max_date + pd.DateOffset(months=mi)
                    future_months.append(fm.strftime("%b %Y"))

                lines.append(f"### {plan_months}-Month Budget Plan ({future_months[0]} → {future_months[-1]})")
                lines.append("")

                # Build table header
                hdr_cols = ["Category", "3-Mo Avg", "Last Month"] + future_months + ["Flag"]
                lines.append("| " + " | ".join(hdr_cols) + " |")
                lines.append("|" + "|".join(["----------"] * len(hdr_cols)) + "|")

                total_per_month = [0.0] * plan_months

                for cat, avg_amt in top_budget_cats.items():
                    cat_label = friendly_category(cat)
                    avg_val = round(avg_amt, 0)
                    last_val = round(cat_last_mo.get(cat, 0), 0)

                    is_essential = any(e in cat.lower() for e in _essential)
                    overspent = last_val > avg_val * 1.15 if avg_val > 0 else False
                    underspent = last_val < avg_val * 0.80 if avg_val > 0 else False

                    # Compute per-month targets with gradual adjustment
                    month_targets = []
                    if overspent and not is_essential:
                        # Gradually reduce: start from avg, cut 5% each subsequent month
                        for mi in range(plan_months):
                            t = round(avg_val * (1 - 0.05 * mi), 0)
                            month_targets.append(max(t, round(avg_val * 0.75, 0)))  # floor at 75% of avg
                        flag = "⚠️ Reduce"
                    elif overspent and is_essential:
                        # Essential: hold at average, slight reduction after month 1
                        for mi in range(plan_months):
                            t = round(avg_val * (1 - 0.02 * mi), 0)
                            month_targets.append(max(t, round(avg_val * 0.90, 0)))
                        flag = "⚠️ Over"
                    elif underspent:
                        # Keep good behaviour — maintain last month's lower level
                        for mi in range(plan_months):
                            month_targets.append(round(last_val, 0))
                        flag = "✅ Good"
                    else:
                        # On track — hold steady
                        for mi in range(plan_months):
                            month_targets.append(avg_val)
                        flag = "→ On track"

                    for mi, t in enumerate(month_targets):
                        total_per_month[mi] += t

                    mo_cells = [f"**${t:,.0f}**" for t in month_targets]
                    lines.append(
                        f"| {cat_label} | ${avg_val:,.0f} | ${last_val:,.0f} | "
                        + " | ".join(mo_cells) + f" | {flag} |"
                    )

                # Others row
                others_avg = round(all_cats_3mo.iloc[8:].sum(), 0) if len(all_cats_3mo) > 8 else 0
                others_last = round(sum(cat_last_mo.get(c, 0) for c in all_cats_3mo.index[8:]), 0) if len(all_cats_3mo) > 8 else 0
                if others_avg > 0 or others_last > 0:
                    mo_cells = [f"**${others_avg:,.0f}**"] * plan_months
                    lines.append(
                        f"| Other | ${others_avg:,.0f} | ${others_last:,.0f} | "
                        + " | ".join(mo_cells) + " | |"
                    )
                    for mi in range(plan_months):
                        total_per_month[mi] += others_avg

                # Totals row
                lines.append(
                    "| **Total** | **${:,.0f}** | **${:,.0f}** | ".format(avg_monthly_total, last_mo_total)
                    + " | ".join([f"**${t:,.0f}**" for t in total_per_month]) + " | |"
                )

                # Projected savings per month
                lines.append("")
                lines.append("### Projected Monthly Savings")
                lines.append("")
                lines.append("| Month | Projected Spend | Income | Savings | Rate |")
                lines.append("|-------|----------------|--------|---------|------|")
                for mi in range(plan_months):
                    proj_save = recent_inc - total_per_month[mi]
                    proj_rate = round((proj_save / recent_inc * 100), 1) if recent_inc > 0 else 0
                    save_emoji = "✅" if proj_rate >= 20 else ("⚠️" if proj_rate > 0 else "❌")
                    lines.append(
                        f"| {future_months[mi]} | ${total_per_month[mi]:,.0f} | ${recent_inc:,.0f} | "
                        f"${proj_save:,.0f} | {save_emoji} {proj_rate}% |"
                    )

            lines.append("")

            # Actionable recommendation
            if savings_pct >= 20:
                lines.append(
                    f"You're saving {savings_pct}% — above the 20% target. "
                    f"Keep your total spend under **${target_20_spend:,.0f}** to stay on track."
                )
            elif savings_pct > 0:
                gap = round(target_20_spend - avg_monthly_total, 2)
                lines.append(
                    f"**Action:** Cut **${abs(gap):,.0f}/mo** from ⚠️ flagged categories to hit the 20% savings target."
                )
            else:
                lines.append(
                    f"**Action:** You're spending more than you earn. Start by reducing ⚠️ flagged categories — "
                    f"even a 15% cut there recovers ~${round(sum(cat_3mo_avg.nlargest(3)) * 0.15, 0):,.0f}/mo."
                )

            if plan_months > 1:
                # Show category trends + income vs expense for multi-month context
                _try_chart("plot_category_trends", months=6)
                _try_chart("plot_income_vs_expense", months=6)
            else:
                _try_chart("plot_income_vs_expense", months=3)
                _try_chart("plot_savings_ratio", months=3)

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
            period_note = f" ({period_label})" if period is not None else ""
            if not top_cats.empty:
                top_pct = (top_cats.iloc[0] / total_exp * 100) if total_exp else 0
                top3_pct = sum((v / total_exp * 100) for v in top_cats.iloc[:3].values) if total_exp else 0
                lines.append(
                    f"Your biggest spending category{period_note} is **{friendly_category(top_cats.index[0])}** at ${top_cats.iloc[0]:,.2f} "
                    f"({top_pct:.1f}% of your spending{period_note}). Your top 3 categories account for {top3_pct:.0f}% of spending:"
                )
                for cat, amt in top_cats.items():
                    pct = (amt / total_exp * 100) if total_exp else 0
                    lines.append(f"  • {friendly_category(cat)}: ${amt:,.2f} ({pct:.1f}%)")
                if top_pct > 50:
                    lines.append(
                        f"\nWorth noting: {friendly_category(top_cats.index[0])} alone accounts for more than half your spending. "
                        f"This is often a fixed cost (like rent or loan payments), but it's worth reviewing if there's flexibility."
                    )
            if _chart_months:
                _try_chart("plot_category_breakdown", months=_chart_months)
            else:
                _try_chart("plot_category_breakdown")
            _try_chart("plot_top_merchants", **{"months": _chart_months} if _chart_months else {})

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

        elif (any(w in prompt_lower for w in ["spend", "expens", "expenditure", "cost", "bought", "paid",
                                              "purchase", "pay", "bill"])
              and not any(w in prompt_lower for w in ["subscription", "subscriptions", "recurring", "streaming"])):
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

        elif (any(w in prompt_lower for w in ["graph", "chart", "plot", "visual", "show me", "display",
                                              "draw", "diagram"])
              and not any(w in prompt_lower for w in ["subscription", "subscriptions", "recurring"])):
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

        elif (any(w in prompt_lower for w in ["how much", "total", "amount", "sum"])
              and not any(w in prompt_lower for w in ["subscription", "subscriptions", "recurring", "streaming"])):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            period_note = f" ({period_label})" if period is not None else ""
            lines.append(
                f"Your totals{period_note}: ${total_exp:,.2f} spent, ${total_inc:,.2f} earned — a net of ${net_savings:+,.2f} "
                f"across {len(expenses):,} transactions ({savings_rate:.1f}% savings rate)."
            )
            _try_chart("plot_income_vs_expense", **{"months": _chart_months} if _chart_months else {})

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

        elif any(w in prompt_lower for w in [
            "subscription", "subscriptions", "recurring", "recurring charge",
            "cancel", "cancellation", "pause", "streaming", "membership",
            "netflix", "spotify", "hulu", "disney", "apple music", "youtube premium",
            "gym", "auto-pay", "autopay", "monthly charge", "recurring fee",
            "what am i subscribed to", "what subscriptions",
        ]) or re.search(r"(recurring|subscription|auto.?pay).*(charge|fee|payment|bill)", prompt_lower):
            # ── Subscription / Recurring Charge Detection ──
            # Smart detection: separate subscriptions from regular bills
            if not expenses.empty:
                exp_with_month = expenses.copy()
                exp_with_month["month"] = exp_with_month["transaction_date"].dt.to_period("M")

                # Categories that are bills/utilities, NOT subscriptions
                _bill_categories = {
                    "rent", "housing", "mortgage", "insurance", "utility", "utilities",
                    "electric", "gas", "water", "phone", "internet", "telecom",
                    "loan", "auto", "car", "medical", "health", "tax",
                }
                # Categories likely to contain subscriptions
                _sub_categories = {
                    "subscription", "entertainment", "streaming", "digital", "software",
                    "media", "music", "gaming", "fitness", "gym", "cloud", "storage",
                }

                merchant_months = exp_with_month.groupby(["merchant_name", "transaction_category_detail"]).agg(
                    month_count=("month", "nunique"),
                    total=("transaction_amount", "sum"),
                    avg_amount=("transaction_amount", "mean"),
                    std_amount=("transaction_amount", "std"),
                    txn_count=("transaction_amount", "count"),
                ).reset_index()
                merchant_months["std_amount"] = merchant_months["std_amount"].fillna(0)

                # Recurring = 2+ months AND consistent amounts (std < 35% of mean)
                recurring = merchant_months[
                    (merchant_months["month_count"] >= 2) &
                    (merchant_months["std_amount"] <= merchant_months["avg_amount"] * 0.35)
                ].copy()

                if not recurring.empty:
                    # Classify each recurring charge
                    def _classify(row):
                        cat = str(row["transaction_category_detail"]).lower()
                        avg = row["avg_amount"]
                        cat_parts = set(cat.replace("_", " ").split())
                        if cat_parts & _bill_categories or avg > 200:
                            return "bill"
                        elif cat_parts & _sub_categories or avg <= 50:
                            return "subscription"
                        elif avg <= 100:
                            return "subscription"
                        else:
                            return "bill"

                    recurring["type"] = recurring.apply(_classify, axis=1)

                    subs = recurring[recurring["type"] == "subscription"].sort_values("avg_amount", ascending=False)
                    bills = recurring[recurring["type"] == "bill"].sort_values("avg_amount", ascending=False)

                    monthly_income = total_inc / max(num_months, 1)

                    # ── Subscriptions Section ──
                    if not subs.empty:
                        total_subs = subs["avg_amount"].sum()
                        sub_pct = (total_subs / monthly_income * 100) if monthly_income > 0 else 0

                        lines.append(
                            f"I found **{len(subs)} subscription(s)** totaling approximately "
                            f"**${total_subs:,.2f}/month** ({sub_pct:.1f}% of monthly income).\n"
                        )
                        lines.append("| Service | Monthly Cost | Active Months | Status |")
                        lines.append("|---|---|---|---|")

                        flagged_count = 0
                        flagged_total = 0.0
                        for _, row in subs.head(15).iterrows():
                            avg = row["avg_amount"]
                            months_active = int(row["month_count"])
                            merch = row["merchant_name"]
                            if avg > 30:
                                flag = "⚠️ Premium — review"
                                flagged_count += 1
                                flagged_total += avg
                            elif months_active >= 8 and avg > 10:
                                flag = "🔍 Long-running — still using?"
                                flagged_count += 1
                                flagged_total += avg
                            elif avg < 5:
                                flag = "✅ Low cost"
                            else:
                                flag = "✅ Active"
                            lines.append(f"| {merch} | ${avg:,.2f} | {months_active} | {flag} |")

                        lines.append("")

                        if flagged_count > 0:
                            lines.append(
                                f"**{flagged_count} subscription(s) flagged for review** — potential savings of "
                                f"up to **${flagged_total:,.2f}/month** (${flagged_total * 12:,.2f}/year) "
                                f"if you cancel or downgrade.\n"
                            )

                        # Actionable advice
                        if sub_pct > 8:
                            lines.append(
                                f"At **{sub_pct:.1f}%** of income, your subscriptions are above the "
                                f"recommended 5-8%. Consider cancelling services you haven't used recently."
                            )
                        elif sub_pct > 5:
                            lines.append(
                                f"Your subscription spend at **{sub_pct:.1f}%** of income is within range, "
                                f"but worth reviewing for unused services."
                            )
                        else:
                            lines.append(
                                f"At **{sub_pct:.1f}%** of income, your subscriptions are well-controlled."
                            )
                    else:
                        lines.append(
                            "No clear subscription-type recurring charges found in your data."
                        )

                    # ── Regular Bills Section (brief) ──
                    if not bills.empty:
                        total_bills = bills["avg_amount"].sum()
                        lines.append(
                            f"\n**Regular recurring bills** ({len(bills)} detected, ~${total_bills:,.2f}/month):"
                        )
                        for _, row in bills.head(8).iterrows():
                            cat_label = friendly_category(row["transaction_category_detail"])
                            lines.append(
                                f"  • {row['merchant_name']} ({cat_label}): ${row['avg_amount']:,.2f}/mo"
                            )

                    # Vola product tie-in
                    lines.append(
                        f"\nVola's **Subscription Tracker** can monitor these automatically — it flags charges "
                        f"you haven't used recently, lets you cancel or pause directly from the app, and "
                        f"you can set custom alerts (e.g. \"notify me if a subscription exceeds $15\")."
                    )

                    # If user is asking for advice (should I, how much, afford), send to LLM for curated advice
                    if re.search(r"(should|afford|how much|recommend|worth|start|new|add)", prompt_lower):
                        monthly_income = total_inc / max(num_months, 1)
                        sub_data = "\n".join(
                            f"- {row['merchant_name']}: ${row['avg_amount']:,.2f}/mo ({int(row['month_count'])} months)"
                            for _, row in subs.head(10).iterrows()
                        ) if not subs.empty else "No subscriptions detected"
                        try:
                            advice_result = self.llm.chat(messages=[
                                {"role": "system", "content": (
                                    "You are Vola AI, a smart financial assistant. The member is asking about "
                                    "subscription management. You have their actual subscription data and financial "
                                    "profile. Give specific, actionable advice. Reference the 5-8% of income "
                                    "guideline for subscriptions. If they ask about adding a new service, calculate "
                                    "how much room they have. Keep it under 80 words. Be direct and friendly. "
                                    "Use bold for key numbers."
                                )},
                                {"role": "user", "content": (
                                    f"Member's question: {prompt}\n\n"
                                    f"Monthly income: ${monthly_income:,.2f}\n"
                                    f"Current subscription total: ${total_subs if not subs.empty else 0:,.2f}/mo\n"
                                    f"Subscription % of income: {sub_pct:.1f}%\n"
                                    f"Savings rate: {(net_savings / total_inc * 100) if total_inc else 0:.1f}%\n"
                                    f"Current subscriptions:\n{sub_data}"
                                )},
                            ])
                            if not advice_result.get("error") and advice_result.get("content", "").strip():
                                lines.append(f"\n**Personalized advice:**\n{self._clean_response(advice_result['content'].strip())}")
                        except Exception:
                            pass  # Silent fail — data table already provides value
                else:
                    lines.append(
                        f"I didn't find clearly recurring charges in your data — your transactions "
                        f"show mostly variable spending patterns. No hidden subscription fees draining your account."
                    )
                    lines.append(
                        f"\nVola's Subscription Tracker can continuously scan your linked accounts for new "
                        f"recurring charges as they appear."
                    )
            else:
                lines.append("No expense data available to analyze for subscriptions.")

            _try_chart("plot_top_merchants")
            _try_chart("plot_category_breakdown")

        elif any(w in prompt_lower for w in ["hello", "hi ", "hi!", "hey", "greet", "good morning",
                                              "good afternoon", "good evening", "good day", "howdy",
                                              "sup", "greetings"]) or prompt_lower.strip() in ("hi", "hey", "yo", "hello"):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            lines.append(
                f"Hey {user_name}! I'm **Vola AI**, your smart financial assistant. "
                f"I've got your complete transaction history loaded — "
                f"**{len(user_df):,} transactions** spanning {num_months} months.\n"
            )
            lines.append(f"**Quick financial snapshot:**")
            lines.append(f"  • Total earned: **${total_inc:,.2f}**")
            lines.append(f"  • Total spent: **${total_exp:,.2f}**")
            lines.append(f"  • Net position: **${net_savings:+,.2f}** "
                         f"({'saving {:.1f}%'.format(savings_rate) if savings_rate > 0 else 'deficit of {:.1f}%'.format(abs(savings_rate))})")
            if not top_cats.empty:
                lines.append(f"  • Biggest expense: **{friendly_category(top_cats.index[0])}** "
                             f"(${top_cats.iloc[0]:,.2f})")
            lines.append(
                f"\nAsk me anything — spending trends, category breakdowns, merchant analysis, "
                f"savings advice, or just say **\"show me a chart\"** to get started!"
            )

        elif any(w in prompt_lower for w in ["thank", "thanks", "appreciate", "great", "awesome", "perfect",
                                              "nice", "good job", "well done", "cool", "ok", "okay",
                                              "got it", "understood", "noted"]):
            lines.append(
                f"You're welcome, {user_name}! Here are some ideas for what to explore next:\n"
            )
            lines.append("  • **\"Show my spending trends\"** — see how your spending changes month to month")
            lines.append("  • **\"Am I saving enough?\"** — check your savings rate against benchmarks")
            lines.append("  • **\"Break down my food spending\"** — deep-dive into any category")
            lines.append("  • **\"Give me a full financial report\"** — the complete picture with charts")
            lines.append(
                f"\nI'm here whenever you need — just ask!"
            )

        elif (any(w in prompt_lower for w in ["help", "what can you", "what do you", "what are you",
                                               "who are you", "options", "commands", "what else",
                                               "functionalities", "functionality", "capabilities",
                                               "capability", "features", "abilities", "scope",
                                               "purpose", "what is vola", "what is this",
                                               "introduce yourself", "tell me about you",
                                               "how does this work", "how do you work",
                                               "what should i ask", "what kind of",
                                               "show me what you can", "what do you know"])
              or re.search(r"what\s+(can|do)\s+you", prompt_lower)
              or re.search(r"how\s+(can|do)\s+you\s+help", prompt_lower)
              or re.search(r"what\s+is\s+your\s+(scope|role|job|purpose)", prompt_lower)
              or re.search(r"what\s+are\s+(your|the)\s+(feature|function|capabilit|option|skill)", prompt_lower)):
            savings_rate = (net_savings / total_inc * 100) if total_inc else 0
            lines.append(
                f"I'm **Vola AI**, your smart financial assistant built into the Vola Finance platform. I have direct access to your complete "
                f"transaction history ({len(user_df):,} transactions across {num_months} months) and I can turn "
                f"that raw data into actionable financial intelligence.\n"
            )
            lines.append("Here's everything I can do for you:\n")
            lines.append("**Spending Analysis**")
            lines.append("  • \"What did I spend the most on?\" — top categories ranked by total")
            lines.append("  • \"Break down my expenses\" — full category breakdown with percentages")
            lines.append("  • \"How much did I spend on food?\" — deep-dive into any specific category")
            lines.append("  • \"Where do I shop the most?\" — merchant analysis ranked by spend\n")
            lines.append("**Trends & Patterns**")
            lines.append("  • \"Show my spending trends\" — month-by-month changes with rolling averages")
            lines.append("  • \"Which day do I spend the most?\" — weekday spending patterns")
            lines.append("  • \"How has my spending changed over time?\" — long-term trajectory\n")
            lines.append("**Income & Savings**")
            lines.append("  • \"Am I saving enough?\" — savings rate vs. the 20% benchmark")
            lines.append("  • \"Compare my income and expenses\" — month-by-month income vs. outflow")
            lines.append("  • \"How much did I earn?\" — income breakdown by source\n")
            lines.append("**Budgeting & Advice**")
            lines.append("  • \"How should I budget next month?\" — realistic budget based on your actual data")
            lines.append("  • \"Can I afford to save more?\" — actionable saving opportunities")
            lines.append("  • \"Where can I cut back?\" — identify discretionary spending to reduce\n")
            lines.append("**Visualizations (12 chart types)**")
            lines.append("  • \"Show me a chart\" — I'll pick the most relevant visualization")
            lines.append("  • \"Show my category breakdown\" — donut chart of where your money goes")
            lines.append("  • \"Plot my spending trend\" — line chart with rolling average")
            lines.append("  • \"Show a heatmap\" — category × month spending intensity grid\n")
            lines.append("**Reports & Deep-Dives**")
            lines.append("  • \"Give me a full financial report\" — complete overview with multiple charts")
            lines.append("  • \"Financial health check\" — savings rate, benchmarks, and verdict")
            lines.append("  • Ask about any **specific merchant** or **category** by name\n")
            lines.append("**Follow-ups**")
            lines.append("  • \"Tell me more\" / \"Explain that\" — get a deeper narrative explanation")
            lines.append("  • \"Show a better chart\" — regenerate with different parameters")
            lines.append("  • \"What does that mean?\" — plain-English interpretation of the data")
            lines.append(
                f"\nJust type your question in plain English — I'll figure out the rest. "
                f"Try starting with: **\"What did I spend the most on?\"**"
            )

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

        # ── Append a contextual follow-up question ──
        followup_q = self._generate_followup_question(prompt_lower, chart_paths, top_cats, total_exp, total_inc, net_savings)
        if followup_q:
            lines.append(f"\n{followup_q}")

        return "\n".join(lines), chart_paths

    def _generate_followup_question(self, prompt_lower: str, chart_paths: list,
                                     top_cats, total_exp: float, total_inc: float,
                                     net_savings: float) -> str:
        """Generate a contextual follow-up question based on what was just discussed."""

        # Map the topic of the current response to relevant follow-up options
        savings_rate = (net_savings / total_inc * 100) if total_inc else 0
        top_cat_name = friendly_category(top_cats.index[0]) if hasattr(top_cats, 'index') and not top_cats.empty else "your top category"

        if any(w in prompt_lower for w in ["budget", "plan", "recommend", "advice", "suggest"]):
            return (
                "What would you like to explore next?\n"
                f"  1. Drill into **{top_cat_name}** spending by merchant\n"
                "  2. See my spending trend over the last 6 months\n"
                "  3. Show my recurring and subscription charges"
            )
        elif any(w in prompt_lower for w in ["subscription", "recurring", "streaming", "cancel"]):
            return (
                "Would you like me to:\n"
                "  1. Build a budget plan to help cut costs\n"
                "  2. Show how these charges compare to my overall spending\n"
                "  3. Analyze all recurring charges month-by-month"
            )
        elif any(w in prompt_lower for w in ["trend", "over time", "month", "pattern"]):
            return (
                f"Want to dig deeper?\n"
                f"  1. Break down the trend by category to see what's driving changes\n"
                f"  2. Compare your income vs expenses side by side\n"
                f"  3. Get a budget recommendation for next month"
            )
        elif any(w in prompt_lower for w in ["category", "breakdown", "where", "split", "donut"]):
            return (
                f"Would you like me to drill into **{top_cat_name}** and show a merchant-by-merchant breakdown?"
            )
        elif any(w in prompt_lower for w in ["merchant", "store", "shop", "vendor"]):
            return (
                "Want to see more?\n"
                "  1. Show my spending trend at this merchant over time\n"
                "  2. Compare merchant spending across categories\n"
                "  3. Get a full category breakdown of my spending"
            )
        elif any(w in prompt_lower for w in ["income", "earn", "salary"]):
            if savings_rate < 20:
                return "Your savings rate is below 20%. Would you like me to build a budget plan to improve it?"
            else:
                return (
                    "What would you like to explore next?\n"
                    "  1. See my spending breakdown by category\n"
                    "  2. View my monthly savings trend\n"
                    "  3. Get a full financial report"
                )
        elif any(w in prompt_lower for w in ["saving", "save", "savings"]):
            return (
                "Would you like me to:\n"
                "  1. Identify which categories have the most room to cut\n"
                f"  2. Drill into **{top_cat_name}** to find savings opportunities\n"
                "  3. Create a detailed budget plan for next month"
            )
        elif any(w in prompt_lower for w in ["spend", "expens", "cost", "paid", "purchase"]):
            return (
                "Want to explore further?\n"
                "  1. See the month-by-month spending trend\n"
                f"  2. Drill into **{top_cat_name}** by merchant\n"
                "  3. Compare income vs expenses"
            )
        elif any(w in prompt_lower for w in ["compare", "vs", "versus", "income and expense"]):
            return "Would you like me to create a budget plan based on these numbers to optimize your savings?"
        elif any(w in prompt_lower for w in ["report", "overview", "everything", "summary"]):
            return (
                "What would you like to focus on?\n"
                "  1. Deep-dive into my top spending category\n"
                "  2. Analyze my subscriptions and recurring charges\n"
                "  3. Build a budget plan for next month"
            )
        elif any(w in prompt_lower for w in ["highest", "top", "biggest", "most", "maximum"]):
            return f"Would you like me to break down **{top_cat_name}** by merchant to see where exactly that money is going?"
        elif any(w in prompt_lower for w in ["hello", "hi", "hey", "help", "what can"]):
            return (
                "What can I help you with today?\n"
                "  1. Show me a spending overview\n"
                "  2. Analyze my income vs expenses\n"
                "  3. Build a budget plan for next month"
            )
        else:
            # Generic follow-up for any other topic
            return (
                "What would you like to explore next?\n"
                "  1. See my spending trends over time\n"
                "  2. Get a category-by-category breakdown\n"
                "  3. Build a budget plan for next month"
            )

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
            months = kwargs.get("months") or period_map.get(period, 3)
            expenses = _filter(all_expenses, months)
            if months == 1:
                period_label = "last month"
            else:
                period_label = f"last {months} months"
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

        # Resolve short follow-up replies (yes/no/1/2/3) into full queries
        processed_prompt, is_followup = self._resolve_followup(processed_prompt, user_id)
        if is_followup:
            flags.append("followup_resolved")

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
        chart_paths: list = []

        # ── Intercept: comprehensive report bypasses LLM for deterministic output ──
        if not is_followup and self._is_full_report_intent(processed_prompt):
            response_text, chart_paths = self._comprehensive_report(user_df, user_id)
            flags.append("comprehensive_report")
        elif not is_followup and any(w in processed_prompt.lower() for w in [
            "better graph", "better chart", "improve", "redo", "regenerate",
            "different chart", "another graph", "new graph", "new chart",
            "clearer", "detailed graph", "detailed chart",
        ]):
            # Route "better graph" follow-ups to fallback handler where viz_state-aware logic lives
            response_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            flags.append("better_chart_followup")
        elif not is_followup and (any(w in processed_prompt.lower() for w in [
            "subscription", "subscriptions", "recurring", "cancel",
            "streaming", "autopay", "auto-pay", "recurring charge",
            "what am i subscribed", "recurring fee",
        ]) or re.search(r"(recurring|subscription|auto.?pay).*(charge|fee|payment|bill)", processed_prompt.lower()) \
           or re.search(r"(afford|should|how much).*(subscription|streaming|recurring)", processed_prompt.lower())):
            # Route subscription queries to fallback where recurring charge detection lives
            response_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            flags.append("subscription_analysis")
        elif not is_followup and any(w in processed_prompt.lower() for w in [
            "budget", "how should i", "plan next month", "recommend",
            "advice", "suggest", "what should i spend", "can i afford", "reduce", "cut back",
        ]):
            # Route budget/advice queries to fallback where prescriptive table logic lives
            response_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            flags.append("budget_advice")
        elif (llm_result := self.llm.chat(messages=messages, tools=TOOL_SCHEMAS)).get("error"):
            response_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            flags.append("llm_fallback")
        else:
            response_text = llm_result.get("content", "") or ""
            tool_calls = llm_result.get("tool_calls") or []

            if tool_calls:
                chart_paths = self._execute_tool_calls(tool_calls, user_id, prompt=processed_prompt)

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
                        "content": json.dumps({"status": "success", "charts": [
                            p["path"] if isinstance(p, dict) else p for p in chart_paths
                        ]}),
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
        self.cache.add_conversation_turn(user_id, "assistant", response_text[:2000])
        # Update viz state if charts were generated
        if chart_paths:
            last_chart_item = chart_paths[-1]
            if isinstance(last_chart_item, dict):
                chart_type = last_chart_item.get("fn", "")
                chart_kwargs = last_chart_item.get("kwargs", {})
                if not chart_type:
                    last_path = last_chart_item.get("path", "")
                    for fn_name in VIZ_FUNCTIONS:
                        if fn_name in last_path:
                            chart_type = fn_name
                            break
            else:
                chart_type = ""
                chart_kwargs = {}
                for fn_name in VIZ_FUNCTIONS:
                    if fn_name in last_chart_item:
                        chart_type = fn_name
                        break
            self.cache.set_viz_state(user_id, chart_type, chart_kwargs, chart_kwargs)

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
                {
                    "path": p["path"] if isinstance(p, dict) else p,
                    "explanation": self._explain_chart(p, user_df),
                }
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

        # Resolve short follow-up replies (yes/no/1/2/3) into full queries
        processed_prompt, is_followup = self._resolve_followup(processed_prompt, user_id)
        if is_followup:
            flags.append("followup_resolved")

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
        chart_paths: list = []

        # ── Intercept: comprehensive report bypasses LLM for deterministic output ──
        if not is_followup and self._is_full_report_intent(processed_prompt):
            report_text, chart_paths = self._comprehensive_report(user_df, user_id)
            flags.append("comprehensive_report")
            # Stream the report text in chunks to simulate typing
            words = report_text.split(" ")
            chunk = ""
            first_chunk = True
            for i, word in enumerate(words):
                chunk += (" " if chunk else "") + word
                if len(chunk) >= 50 or i == len(words) - 1:
                    text_to_send = chunk if first_chunk else " " + chunk
                    yield {"event": "chunk", "data": {"text": text_to_send}}
                    chunk = ""
                    first_chunk = False
            full_text = report_text
        elif not is_followup and any(w in processed_prompt.lower() for w in [
            "better graph", "better chart", "improve", "redo", "regenerate",
            "different chart", "another graph", "new graph", "new chart",
            "clearer", "detailed graph", "detailed chart",
        ]):
            # Route "better graph" follow-ups to fallback handler
            flags.append("better_chart_followup")
            fallback_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            words = fallback_text.split(" ")
            chunk = ""
            first_chunk = True
            for i, word in enumerate(words):
                chunk += (" " if chunk else "") + word
                if len(chunk) >= 30 or i == len(words) - 1:
                    text_to_send = chunk if first_chunk else " " + chunk
                    yield {"event": "chunk", "data": {"text": text_to_send}}
                    chunk = ""
                    first_chunk = False
            full_text = fallback_text
        elif not is_followup and (any(w in processed_prompt.lower() for w in [
            "subscription", "subscriptions", "recurring", "cancel",
            "streaming", "autopay", "auto-pay", "recurring charge",
            "what am i subscribed", "recurring fee",
        ]) or re.search(r"(recurring|subscription|auto.?pay).*(charge|fee|payment|bill)", processed_prompt.lower()) \
           or re.search(r"(afford|should|how much).*(subscription|streaming|recurring)", processed_prompt.lower())):
            # Route subscription queries to fallback where recurring charge detection lives
            flags.append("subscription_analysis")
            fallback_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            words = fallback_text.split(" ")
            chunk = ""
            first_chunk = True
            for i, word in enumerate(words):
                chunk += (" " if chunk else "") + word
                if len(chunk) >= 30 or i == len(words) - 1:
                    text_to_send = chunk if first_chunk else " " + chunk
                    yield {"event": "chunk", "data": {"text": text_to_send}}
                    chunk = ""
                    first_chunk = False
            full_text = fallback_text
        elif not is_followup and any(w in processed_prompt.lower() for w in [
            "budget", "how should i", "plan next month", "recommend",
            "advice", "suggest", "what should i spend", "can i afford", "reduce", "cut back",
        ]):
            # Route budget/advice queries to fallback where prescriptive table logic lives
            flags.append("budget_advice")
            fallback_text, chart_paths = self._fallback_response(
                user_df, processed_prompt, user_id, conversation=conversation
            )
            words = fallback_text.split(" ")
            chunk = ""
            first_chunk = True
            for i, word in enumerate(words):
                chunk += (" " if chunk else "") + word
                if len(chunk) >= 30 or i == len(words) - 1:
                    text_to_send = chunk if first_chunk else " " + chunk
                    yield {"event": "chunk", "data": {"text": text_to_send}}
                    chunk = ""
                    first_chunk = False
            full_text = fallback_text
        else:
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
                    chart_paths = self._execute_tool_calls(tool_calls, user_id, prompt=processed_prompt)

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
        self.cache.add_conversation_turn(user_id, "assistant", full_text[:2000])
        # Update viz state if charts were generated
        if chart_paths:
            last_chart_item = chart_paths[-1]
            if isinstance(last_chart_item, dict):
                chart_type = last_chart_item.get("fn", "") or ""
                chart_kwargs = last_chart_item.get("kwargs", {})
                if not chart_type:
                    last_path = last_chart_item.get("path", "")
                    for fn_name in VIZ_FUNCTIONS:
                        if fn_name in last_path:
                            chart_type = fn_name
                            break
            else:
                chart_type = ""
                chart_kwargs = {}
                for fn_name in VIZ_FUNCTIONS:
                    if fn_name in last_chart_item:
                        chart_type = fn_name
                        break
            self.cache.set_viz_state(user_id, chart_type, chart_kwargs, chart_kwargs)

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
