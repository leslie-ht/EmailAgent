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

Plus one language-agnostic fallback (`detect_non_english`): categories 3 and
4 above are English-keyword regexes, so a non-English email can otherwise
walk straight around them by language choice alone rather than by evading
the pattern. When a large share of the email looks non-English, this file
forces a minimum of ASK_FIRST regardless of whether any regex matched --
see `detect_non_english` for why, and DESIGN.md §7 for the residual
limitation this doesn't solve (it can't tell a risky non-English email from
a benign one, only that it can't tell).
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


# Fraction of alphabetic characters that must be non-ASCII before we treat an
# email as "likely non-English" for the purposes of this fallback. This is
# deliberately crude (no real language detection dependency) -- it exists
# only to catch the case where MONEY_PATTERNS / INJECTION_PATTERNS, being
# English-keyword regexes, simply cannot match at all, which would otherwise
# let a non-English money-flavored or injection email sail through with zero
# reasons and PROCEED_SILENT.
_NON_ENGLISH_ALPHA_RATIO = 0.3


def detect_non_english(email: Email) -> bool:
    """
    Best-effort, language-agnostic signal that the regex patterns above
    cannot be trusted for this email: if a large share of its alphabetic
    characters are outside ASCII, none of the English-keyword regexes had a
    realistic chance of matching, so "no pattern hit" must NOT be read as
    "no risk." This is intentionally crude -- it does not attempt to
    identify the language, translate, or otherwise interpret the content
    (see the module docstring: nothing in this file should grow an LLM
    dependency). It only decides whether the absence of a regex match is
    meaningful evidence of safety.
    """
    text = f"{email.subject}{email.body}"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    return (non_ascii / len(letters)) > _NON_ENGLISH_ALPHA_RATIO


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

    if detect_non_english(email):
        reasons.append(
            "content is likely non-English -- the money/injection regexes above are "
            "English-keyword-only and cannot be trusted to have caught anything, so a "
            "conservative floor applies regardless of whether they matched"
        )
        floor = tightest(floor, Decision.ASK_FIRST)

    is_gated = len(reasons) > 0
    return SafetyCheckResult(is_gated=is_gated, min_decision=floor, reasons=reasons)
