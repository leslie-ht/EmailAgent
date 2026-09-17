"""
Core data schemas shared across the pipeline.

Design note: these are deliberately plain dataclasses (no ORM, no pydantic
dependency) so the whole agent can run with zero external deps beyond an
optional LLM SDK. Keeping the schema small and explicit also makes the
safety gate's job auditable: it only ever looks at fields defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StakesTier(str, Enum):
    LOW = "low"          # reversible, internal, no money, no external send
    MEDIUM = "medium"     # reversible but visible/consequential
    HIGH = "high"          # irreversible, external, money, or injection risk


class ActionType(str, Enum):
    ARCHIVE = "archive"
    LABEL = "label"
    DELETE = "delete"                    # irreversible -> always gated
    UNSUBSCRIBE = "unsubscribe"           # often irreversible -> gated
    DRAFT_REPLY = "draft_reply"            # reversible (draft only)
    SEND_REPLY = "send_reply"               # irreversible once sent
    FORWARD = "forward"                      # external -> gated
    SCHEDULE_MEETING = "schedule_meeting"     # external -> gated
    FLAG_URGENT = "flag_urgent"                 # internal, reversible
    NO_ACTION = "no_action"


class Decision(str, Enum):
    PROCEED_SILENT = "proceed_silent"
    PROCEED_NOTIFY = "proceed_notify"
    ASK_FIRST = "ask_first"
    ESCALATE = "escalate"


@dataclass
class Email:
    id: str
    sender: str                 # email address
    sender_domain: str
    subject: str
    body: str
    known_sender: bool = False    # has the user corresponded with this address before
    thread_has_prior_context: bool = False
    ground_truth_category: Optional[str] = None       # for eval only
    ground_truth_decision: Optional[Decision] = None    # for eval only


@dataclass
class ClassificationResult:
    category: str                     # e.g. "newsletter", "invoice", "meeting_request"
    proposed_action: ActionType
    stakes_tier: StakesTier
    confidence: float                  # classifier's own confidence, 0-1
    source: str                          # "llm" or "heuristic"
    injection_flag: bool = False
    injection_evidence: Optional[str] = None
    rationale: str = ""
    # Observability for the LLM path only; stay None on the heuristic path.
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class SafetyCheckResult:
    is_gated: bool                    # True => floor applies, cannot go below ASK
    min_decision: Decision              # the most autonomous decision allowed
    reasons: list[str] = field(default_factory=list)


@dataclass
class PolicyDecisionResult:
    decision: Decision
    bucket: str                          # (category, action_type, sender_trust) key used for learning
    confidence_lower_bound: float
    safety: SafetyCheckResult
    classification: ClassificationResult


@dataclass
class ActionOutcome:
    email_id: str
    action_taken: ActionType
    decision: Decision
    log_message: str


@dataclass
class Feedback:
    email_id: str
    bucket: str
    approved: bool                   # True = user approved / correct, False = rejected or corrected
    corrected: bool = False            # True if user modified rather than outright rejected
    note: str = ""
