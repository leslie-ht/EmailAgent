"""
Executor: simulates carrying out an action. Deliberately mocked -- no real
IMAP/SMTP calls -- both for take-home safety (no risk of actually sending
mail) and to keep the eval harness deterministic and offline-runnable.

For PROCEED_SILENT / PROCEED_NOTIFY, the action is "executed" (logged).
For ASK_FIRST, the action is only proposed; it executes only after a
Feedback with approved=True is recorded (see main.py / run_eval.py for the
interactive vs. simulated flow).
For ESCALATE, no action is taken at all -- it's handed to the user as-is.
"""

from __future__ import annotations

from agent.schemas import ActionOutcome, Decision, Email, PolicyDecisionResult


def execute(email: Email, result: PolicyDecisionResult) -> ActionOutcome:
    action = result.classification.proposed_action
    decision = result.decision

    if decision == Decision.PROCEED_SILENT:
        msg = f"[SILENT] Executed '{action.value}' on email {email.id} ({email.subject!r}). No notification sent."
    elif decision == Decision.PROCEED_NOTIFY:
        msg = f"[NOTIFY] Executed '{action.value}' on email {email.id} ({email.subject!r}). User notified after the fact."
    elif decision == Decision.ASK_FIRST:
        msg = (f"[ASK] Proposing '{action.value}' on email {email.id} ({email.subject!r}). "
               f"Waiting for user approval before acting.")
    else:  # ESCALATE
        reasons = "; ".join(result.safety.reasons) or "policy uncertainty"
        msg = f"[ESCALATE] Email {email.id} ({email.subject!r}) flagged for human review. Reasons: {reasons}"

    return ActionOutcome(email_id=email.id, action_taken=action, decision=decision, log_message=msg)
