"""Input, output, and operational guardrails for the pipeline."""

import re
import time
import logging
import hashlib
import os
from typing import Optional

from config import (
    MAX_PROMPT_LENGTH,
    MAX_TOKENS_INPUT,
    CIRCUIT_BREAKER_THRESHOLD,
    LOG_DIR,
)

logger = logging.getLogger("guardrails")

# ═══════════════════════════════════════════════════════════════════
#  INPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(the\s+)?system\s+(prompt|message|instructions)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+",
    r"forget\s+(all\s+)?previous",
    r"disregard\s+(all\s+)?previous",
    r"override\s+(all\s+)?instructions",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*/?system\s*>",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"\bdo\s+anything\s+now\b",
]

FINANCIAL_KEYWORDS = [
    "spend", "spent", "spending", "expense", "expenses",
    "income", "earning", "salary", "save", "saving",
    "transaction", "budget", "category", "categories",
    "merchant", "payment", "money", "financial",
    "trend", "month", "monthly", "week", "weekly",
    "report", "summary", "breakdown", "chart", "plot",
    "total", "average", "highest", "lowest", "most", "least",
    "cost", "pay", "paid", "bill", "subscription",
    "food", "rent", "housing", "transport", "health",
    "entertainment", "shopping", "insurance", "gym",
    # Category names (friendly + raw) so users can respond with just a category
    "groceries", "grocery", "coffee", "cafe", "restaurant", "dining",
    "fast food", "fastfood", "clothing", "clothes", "electronics",
    "flights", "flight", "travel", "hotels", "hotel",
    "fuel", "gas", "rideshare", "taxi", "uber", "lyft",
    "pharmacy", "doctor", "medical", "movies", "movie", "cinema",
    "streaming", "courses", "education", "freelance",
    "pet", "pets", "pet supplies", "utilities", "internet",
    "refund", "refunds", "cashback",
    "compare", "vs", "versus", "between", "last", "this",
    "doing", "financially", "overview", "analysis",
    "finance", "finances", "manage", "managing",
    "best way", "tip", "tips", "strategy", "strategies",
    # Follow-up / graph / conversational keywords
    "graph", "visual", "better", "more", "detail", "redo",
    "improve", "another", "different", "again", "elaborate",
    "explain", "help", "advice", "recommend", "suggest",
    "afford", "reduce", "cut", "earn", "received", "credit",
    "subscription", "subscriptions", "recurring", "cancel",
    "streaming", "membership", "autopay", "vola", "cash advance",
    "credit score", "creditmap", "rent reporting", "mobile plan",
]

# Conversational / meta phrases that should ALWAYS pass scope check.
# These are greetings, capability questions, and polite exchanges
# that the pipeline's fallback engine handles with rich responses.
_CONVERSATIONAL_PATTERNS = [
    # Greetings
    r"^hi$", r"^hi[!. ]", r"^hey", r"^hello", r"^good\s+(morning|afternoon|evening|day)",
    r"^howdy", r"^yo$", r"^yo[!. ]", r"^sup$", r"^greetings",
    # Capability / meta questions
    r"what\s+(can|do)\s+you\s+(do|help|offer|know|analyze|support)",
    r"what\s+are\s+(your|the)\s+(features|functionalities|capabilities|functions|abilities|skills|options)",
    r"what\s+is\s+your\s+(scope|purpose|role|job|function)",
    r"how\s+(can|do)\s+you\s+help",
    r"what\s+else\s+can\s+you",
    r"what\s+should\s+i\s+ask",
    r"what\s+kind\s+of\s+(questions|things|queries|analysis)",
    r"show\s+me\s+what\s+you\s+can",
    r"what\s+do\s+you\s+know",
    r"who\s+are\s+you",
    r"what\s+are\s+you",
    r"tell\s+me\s+about\s+(yourself|you)",
    r"introduce\s+yourself",
    r"what\s+is\s+(this|vola|vola\s*ai|vola\s*finance)",
    r"how\s+does\s+this\s+(work|app|tool|platform)",
    # Vola product queries
    r"(should|can)\s+i\s+(get|take|start|subscribe|use)\s+(a\s+)?(vola|subscription|membership|cash\s+advance)",
    r"how\s+much\s+(does|is|for)\s+(vola|subscription|membership|cash\s+advance)",
    r"what\s+(is|are)\s+(vola|cash\s+advance|credit\s*map|credit\s*builder|subscription\s*tracker)",
    r"(tell|know)\s+(me\s+)?about\s+(vola|cash\s+advance|credit|subscription)",
    # Polite exchanges
    r"^thanks?", r"^thank\s+you", r"^appreciate", r"^great$", r"^awesome",
    r"^perfect", r"^nice$", r"^good\s+job", r"^well\s+done",
    r"^ok$", r"^okay$", r"^got\s+it", r"^understood", r"^cool$",
    # Short follow-up replies (yes/no/numbers) — these are continuations of
    # a prior conversation turn where the AI offered options or asked a question.
    r"^yes$", r"^yes[!., ]", r"^yeah", r"^yep", r"^yup", r"^sure", r"^absolutely",
    r"^no$", r"^no[!., ]", r"^nah", r"^nope", r"^not really",
    r"^[1-9]$", r"^option\s*[1-9]", r"^choice\s*[1-9]",
    r"^go\s+(ahead|for\s+it)", r"^do\s+it", r"^please$", r"^please\s+do",
    r"^let'?s\s+(do|go|see|try)", r"^show\s+me", r"^tell\s+me",
    r"^the\s+(first|second|third|1st|2nd|3rd)", r"^both", r"^all\s+of\s+(them|the\s+above)",
    r"^why\s+not", r"^sounds\s+good", r"^that\s+(works|sounds|would\s+be)",
    r"^i'?d\s+(like|love|prefer|want)", r"^definitely", r"^of\s+course",
]

# These only count if paired with a financial keyword above
_HELPER_WORDS = {"how", "what", "when", "where", "show", "give", "tell"}


def check_prompt_injection(prompt: str) -> Optional[str]:
    prompt_lower = prompt.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, prompt_lower):
            return (
                "I'm sorry, but I can't process that request. "
                "I'm designed to help you analyze your financial transactions. "
                "Please ask me something about your spending, income, or financial trends."
            )
    return None


def check_scope(prompt: str) -> Optional[str]:
    prompt_lower = prompt.lower().strip()

    # Always allow conversational / meta queries through — the pipeline
    # fallback engine handles these with rich, persona-driven responses.
    for pattern in _CONVERSATIONAL_PATTERNS:
        if re.search(pattern, prompt_lower):
            return None

    has_financial = False
    has_helper = False
    for kw in FINANCIAL_KEYWORDS:
        if kw in prompt_lower:
            if kw in _HELPER_WORDS:
                has_helper = True
            else:
                has_financial = True
    if has_financial:
        return None
    return (
        "That question doesn't seem related to your financial transactions. "
        "I can help you analyze your spending, income, savings trends, "
        "category breakdowns, and more. What would you like to know?"
    )


def check_length(prompt: str) -> tuple[str, bool]:
    if len(prompt) > MAX_PROMPT_LENGTH:
        return prompt[:MAX_PROMPT_LENGTH], True
    return prompt, False


def check_cross_user_leakage(
    prompt: str, current_user_id: str, all_user_ids: set[str] | None = None,
) -> Optional[str]:
    prompt_lower = prompt.lower()
    _LEAKAGE_MSG = (
        "I can only show you your own financial data. "
        "I'm not able to access other users' information for privacy and security reasons."
    )

    # Check if any OTHER known user ID appears in the prompt
    if all_user_ids:
        for uid in all_user_ids:
            if uid.lower() != current_user_id.lower() and uid.lower() in prompt_lower:
                return _LEAKAGE_MSG

    # Also catch usr_xxx / user_xxx patterns for IDs not yet in dataset
    for uid in re.findall(r"(?:usr|user)_[a-z0-9]+", prompt_lower):
        normalised = uid if uid.startswith("usr_") else "usr_" + uid[5:]
        if normalised != current_user_id:
            return _LEAKAGE_MSG

    cross_patterns = [
        r"(another|other)\s+user",
        r"user[\s_]\w+'?s?\s+(spending|data|transactions|income)",
        r"someone\s+else",
        r"different\s+user",
    ]
    for pat in cross_patterns:
        if re.search(pat, prompt_lower):
            return _LEAKAGE_MSG
    return None


def run_input_guardrails(prompt: str, user_id: str, all_user_ids: set[str] | None = None) -> dict:
    flags: list[str] = []

    injection_msg = check_prompt_injection(prompt)
    if injection_msg:
        return {"blocked": True, "message": injection_msg, "flags": ["prompt_injection"]}

    leakage_msg = check_cross_user_leakage(prompt, user_id, all_user_ids)
    if leakage_msg:
        return {"blocked": True, "message": leakage_msg, "flags": ["cross_user_leakage"]}

    scope_msg = check_scope(prompt)
    if scope_msg:
        return {"blocked": True, "message": scope_msg, "flags": ["off_topic"]}

    processed_prompt, was_truncated = check_length(prompt)
    if was_truncated:
        flags.append("prompt_truncated")

    return {"blocked": False, "prompt": processed_prompt, "flags": flags}


# ═══════════════════════════════════════════════════════════════════
#  OUTPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

def check_hallucination(response_text: str, data_summary: dict) -> list[str]:
    flags = []
    if not data_summary:
        return flags

    # Only check numbers prefixed with $ (monetary claims)
    numbers_in_response = re.findall(r"\$([\d,]+\.?\d*)", response_text)
    if not numbers_in_response:
        return flags

    ground_truth: set[float] = set()

    def _extract(obj):
        try:
            val = float(obj)
            ground_truth.add(round(abs(val), 2))
            return
        except (TypeError, ValueError):
            pass
        if isinstance(obj, dict):
            for v in obj.values():
                _extract(v)
        elif isinstance(obj, list):
            for v in obj:
                _extract(v)

    _extract(data_summary)

    # Also add common derived values (monthly averages, net savings)
    total_exp = float(data_summary.get("total_expenses", 0))
    total_inc = float(data_summary.get("total_income", 0))
    n_months = max(len(data_summary.get("monthly_expenses", {})), 1)
    ground_truth.add(round(total_exp / n_months, 2))
    ground_truth.add(round(total_inc / n_months, 2))
    ground_truth.add(round(total_inc - total_exp, 2))
    ground_truth.add(round((total_inc - total_exp) / n_months, 2))
    # Add per-month net savings
    ground_truth.add(round(total_inc / n_months - total_exp / n_months, 2))

    _extract(data_summary)

    for ns in numbers_in_response:
        try:
            n = float(ns.replace(",", ""))
            if n <= 10:
                continue
            grounded = any(
                abs(n - gt) < max(1, gt * 0.10) for gt in ground_truth  # 10% tolerance
            )
            if not grounded:
                flags.append(f"ungrounded_number:{ns}")
        except ValueError:
            continue
    return flags


def check_confidence(response_text: str) -> list[str]:
    phrases = [
        "i'm not sure", "i am not sure",
        "i don't know", "i do not know",
        "insufficient data", "not enough data",
        "cannot determine", "can't determine",
        "no data available", "no transactions found",
    ]
    for p in phrases:
        if p in response_text.lower():
            return ["low_confidence"]
    return []


def check_toxicity(response_text: str) -> list[str]:
    """Lightweight keyword-based filter for offensive/inappropriate content."""
    _TOXIC_PATTERNS = [
        r"\bfuck\b", r"\bshit\b", r"\bass\b", r"\bbitch\b", r"\bdamn\b",
        r"\bstupid\b", r"\bidiot\b", r"\bdumb\b", r"\bhate\s+you\b",
        r"\bkill\b", r"\bdie\b", r"\bsuicid", r"\bviolence\b",
        r"\bracis[tm]\b", r"\bsexis[tm]\b", r"\bslur\b",
        r"\bnigger\b", r"\bfaggot\b", r"\bretard\b",
    ]
    text_lower = response_text.lower()
    for pat in _TOXIC_PATTERNS:
        if re.search(pat, text_lower):
            return ["toxic_content"]
    return []


def run_output_guardrails(response_text: str, data_summary: dict) -> dict:
    flags: list[str] = []
    flags.extend(check_hallucination(response_text, data_summary))
    flags.extend(check_confidence(response_text))
    flags.extend(check_toxicity(response_text))
    return {"flags": flags, "response": response_text}


# ═══════════════════════════════════════════════════════════════════
#  OPERATIONAL GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

class CircuitBreaker:
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, reset_seconds: int = 60):
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.is_open = True

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        if self.last_failure_time and (time.time() - self.last_failure_time) > self.reset_seconds:
            self.is_open = False
            self.failure_count = 0
            return True
        return False


class AuditLogger:
    def __init__(self, log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(
            os.path.join(log_dir, "audit.log"), encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self._logger = logging.getLogger("audit")
        if not self._logger.handlers:
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log_request(
        self,
        user_id: str,
        prompt: str,
        response_summary: str,
        latency_ms: float,
        guardrail_flags: list,
        cache_hit: bool,
    ):
        uid_hash = hashlib.sha256(user_id.encode()).hexdigest()[:12]
        prompt_short = (prompt[:100] + "...") if len(prompt) > 100 else prompt
        self._logger.info(
            f"user={uid_hash} | prompt={prompt_short} | "
            f"resp_len={len(response_summary)} | latency={latency_ms:.0f}ms | "
            f"flags={guardrail_flags} | cache_hit={cache_hit}"
        )


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def enforce_token_budget(text: str, max_tokens: int = MAX_TOKENS_INPUT) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars] + "\n[Context truncated to fit token budget]"
