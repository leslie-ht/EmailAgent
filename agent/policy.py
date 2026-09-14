"""
Autonomy policy: combines the safety gate's hard floor with a learned
confidence estimate to produce one of the four decisions.

Learning approach: Beta-Bernoulli bandit per bucket, where a bucket is
(category, proposed_action, sender_trust). This is deliberately simple:
- cheap to update online, one email at a time
- gives a natural confidence interval (not just a point estimate), so we
  threshold on the LOWER bound of a (1 - alpha) credible interval rather
  than the mean. This means a bucket needs *sustained* good feedback
  before autonomy increases -- a single lucky approval can't unlock
  silent mode, which is the calibration property the brief asks for
  ("asks less as it gets preferences right").
- rejections are weighted 2x relative to approvals when updating alpha/
  beta, reflecting that over-trusting is a worse failure than
  under-trusting (asymmetric cost).

Thresholds (tunable, documented in DESIGN.md):
  lower_bound >= SILENT_THRESHOLD  -> eligible for PROCEED_SILENT
  lower_bound >= NOTIFY_THRESHOLD  -> eligible for PROCEED_NOTIFY
  otherwise                          -> ASK_FIRST
The safety gate's min_decision is then applied as a floor that can only
tighten this, never loosen it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from agent.schemas import (
    ClassificationResult,
    Decision,
    Email,
    Feedback,
    PolicyDecisionResult,
    SafetyCheckResult,
)
from agent.safety_gate import check as safety_check
from agent.safety_gate import tightest

SILENT_THRESHOLD = 0.85
NOTIFY_THRESHOLD = 0.60
REJECTION_WEIGHT = 2.0
Z = 1.64  # ~90% one-sided lower confidence bound


def sender_trust_bucket(email: Email) -> str:
    return "known" if email.known_sender else "unknown"


def bucket_key(classification: ClassificationResult, email: Email) -> str:
    return f"{classification.category}::{classification.proposed_action.value}::{sender_trust_bucket(email)}"


@dataclass
class BetaState:
    alpha: float = 1.0  # prior: 1 success
    beta: float = 1.0     # prior: 1 failure  (uniform prior -> starts cautious)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def lower_bound(self, z: float = Z) -> float:
        """Normal approximation to a lower confidence bound on the Beta mean.
        Simple and adequate for this use case; documented as an approximation."""
        n = self.alpha + self.beta
        p = self.mean()
        if n <= 0:
            return 0.0
        se = math.sqrt(max(p * (1 - p), 1e-9) / n)
        return max(0.0, p - z * se)

    def update(self, approved: bool):
        if approved:
            self.alpha += 1.0
        else:
            self.beta += REJECTION_WEIGHT


@dataclass
class PolicyState:
    buckets: dict[str, BetaState] = field(default_factory=dict)

    def get(self, key: str) -> BetaState:
        if key not in self.buckets:
            self.buckets[key] = BetaState()
        return self.buckets[key]

    def record_feedback(self, feedback: Feedback):
        state = self.get(feedback.bucket)
        # a correction counts as a partial signal: not fully approved, but
        # not a full rejection either -- treat as approved=False (still
        # needs asking) but without the extra rejection weight penalty.
        if feedback.corrected and feedback.approved is False:
            state.beta += 1.0
        else:
            state.update(feedback.approved)


def decide(email: Email, classification: ClassificationResult, state: PolicyState) -> PolicyDecisionResult:
    safety: SafetyCheckResult = safety_check(email, classification.proposed_action)

    key = bucket_key(classification, email)
    beta_state = state.get(key)
    lower_bound = beta_state.lower_bound()

    if lower_bound >= SILENT_THRESHOLD:
        learned_decision = Decision.PROCEED_SILENT
    elif lower_bound >= NOTIFY_THRESHOLD:
        learned_decision = Decision.PROCEED_NOTIFY
    else:
        learned_decision = Decision.ASK_FIRST

    # Safety floor can only make the decision LESS autonomous, never more.
    final_decision = tightest(learned_decision, safety.min_decision)

    return PolicyDecisionResult(
        decision=final_decision,
        bucket=key,
        confidence_lower_bound=lower_bound,
        safety=safety,
        classification=classification,
    )
