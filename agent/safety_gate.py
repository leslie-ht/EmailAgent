"""
Safety gate — the hard floor.

CRITICAL DESIGN INVARIANT (read this before touching this file):

    The safety gate is the ONLY module allowed to set a ceiling on autonomy
    based on category-of-risk. It runs before the learned policy and its
    output can only be tightened, never loosened, by anything downstream.
    The policy module imports SafetyCheckResult and treats `min_decision`
    as a hard floor: `final_decision <= min_decision` in autonomy-ranking
    terms (ESCALATE is the least autonomous, PROCEED_SILENT the most).

    Nothing in this file reads from PolicyState, the bandit confidence
    store, or any learned parameter. If you find yourself wanting to pass
    a learned value into this module, stop — that would defeat the
    purpose of a floor that "learning can't weaken".

Four hard-coded trigger categories, per the brief:
    1. Irreversible actions (delete, unsubscribe, send)
    2. External-facing actions (new/unknown recipients)
    3. Money-related content
    4. Suspected prompt injection in the email body
"""

from __future__ import annotations

import re

from agent.schemas import ActionType, Decision, Email, SafetyCheckResult

# Actions that can never be undone once executed.
IRREVERSIBLE_ACTIONS = {
    ActionType.DELETE,
    ActionType.SEND_REPLY,
    ActionType.UNSUBSCRIBE,
}

# Actions that touch the outside world (someone other than the user sees
# something as a direct result).
EXTERNAL_ACTIONS = {
    ActionType.SEND_REPLY,
    ActionType.FORWARD,
    ActionType.SCHEDULE_MEETING,
}

MONEY_PATTERNS = [
    r"\binvoice\b", r"\bwire transfer\b", r"\bpayment (is |was )?(due|required|failed)\b",
    r"\brefund\b", r"\bbank (account|details)\b", r"\bpurchase order\b",
    r"\bcredit card\b", r"\bpay(ment)? (now|immediately)\b", r"\$\d",
    r"\bswift code\b", r"\brouting number\b", r"\boverdue balance\b",
    r"\bdue immediately\b",
]

# Patterns suggestive of a prompt-injection attempt embedded in email
# content. Intentionally broad / high-recall: false positives here just
# mean an extra ask/escalate, which is the safe failure mode. False
# negatives are the actual danger.
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) instructions",
    r"disregard (all |any |the )?(previous|prior|above) instructions",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"act as (an?|the) (admin|administrator|system)",
    r"forward this (email|message) to",
    r"reply (immediately )?with (your|the) (password|credentials|api key|otp)",
    r"click(ing)? (here|this link|the link)( (immediately|now|urgently))?",
    r"do not (tell|notify|inform) the user",
    r"this is (an? )?(urgent )?override",
    r"\bAI\b.{0,20}\b(agent|assistant)\b.{0,30}(must|should|please) (proceed|act|comply)",
    r"account has been suspended",
    r"verify your account (now|immediately|today)",
]

_MONEY_RE = re.compile("|".join(MONEY_PATTERNS), re.IGNORECASE)
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def _rank(d: Decision) -> int:
    """Lower rank = less autonomous. Used to enforce 'floor' semantics."""
    order = {
        Decision.ESCALATE: 0,
        Decision.ASK_FIRST: 1,
        Decision.PROCEED_NOTIFY: 2,
        Decision.PROCEED_SILENT: 3,
    }
    return order[d]


def tightest(a: Decision, b: Decision) -> Decision:
    """Return whichever decision is LESS autonomous (the safer one)."""
    return a if _rank(a) <= _rank(b) else b


def detect_injection(email: Email) -> tuple[bool, str | None]:
    text = f"{email.subject}\n{email.body}"
    match = _INJECTION_RE.search(text)
    if match:
        return True, match.group(0)
    return False, None


def detect_money(email: Email) -> bool:
    text = f"{email.subject}\n{email.body}"
    return bool(_MONEY_RE.search(text))


def check(email: Email, proposed_action: ActionType) -> SafetyCheckResult:
    """
    Evaluate the hard safety floor for a given email + proposed action.

    Returns the MOST PERMISSIVE decision the floor allows (min_decision).
    The policy module must never output anything more autonomous than
    this for the given email/action pair.
    """
    reasons: list[str] = []
    floor = Decision.PROCEED_SILENT  # start permissive, tighten below

    if proposed_action in IRREVERSIBLE_ACTIONS:
        reasons.append(f"irreversible action: {proposed_action.value}")
        floor = tightest(floor, Decision.ASK_FIRST)

    if proposed_action in EXTERNAL_ACTIONS and not email.known_sender:
        reasons.append("external action to an unknown/new recipient")
        floor = tightest(floor, Decision.ASK_FIRST)
    elif proposed_action in EXTERNAL_ACTIONS:
        # even to a known sender, external actions are never silent
        reasons.append("external-facing action")
        floor = tightest(floor, Decision.PROCEED_NOTIFY)

    if detect_money(email):
        reasons.append("money-related content detected")
        floor = tightest(floor, Decision.ASK_FIRST)

    injected, evidence = detect_injection(email)
    if injected:
        reasons.append(f"possible prompt injection detected: '{evidence}'")
        floor = tightest(floor, Decision.ESCALATE)

    is_gated = len(reasons) > 0
    return SafetyCheckResult(is_gated=is_gated, min_decision=floor, reasons=reasons)
