"""
Tests for the feedback oracles: RuleOracle's cautious-vs-risky rank
comparison, NoisyRuleOracle's flip behavior, and LLMPersonaOracle's
parse/fallback interaction (same pattern as LLMClassifier -- see
test_classifier.py -- so this deliberately mirrors that file's structure).
"""

from __future__ import annotations

import pytest

from agent.oracle import (
    LLMPersonaOracle,
    NoisyRuleOracle,
    RuleOracle,
)
from agent.schemas import ActionOutcome, ActionType, Decision, Email


def make_email(**kwargs) -> Email:
    defaults = dict(
        id="e1", sender="a@b.com", sender_domain="b.com",
        subject="Hello", body="Just checking in.", known_sender=True,
    )
    defaults.update(kwargs)
    return Email(**defaults)


def make_outcome(decision: Decision, action: ActionType = ActionType.ARCHIVE) -> ActionOutcome:
    return ActionOutcome(email_id="e1", action_taken=action, decision=decision, log_message="log")


# ---------------------------------------------------------------------------
# RuleOracle
# ---------------------------------------------------------------------------

def test_no_ground_truth_treated_as_neutral_approval():
    email = make_email(ground_truth_decision=None)
    outcome = make_outcome(Decision.PROCEED_SILENT)
    feedback = RuleOracle().give_feedback(email, outcome, bucket="b")
    assert feedback.approved is True


def test_decision_matching_ground_truth_is_approved():
    email = make_email(ground_truth_decision=Decision.PROCEED_NOTIFY)
    outcome = make_outcome(Decision.PROCEED_NOTIFY)
    feedback = RuleOracle().give_feedback(email, outcome, bucket="b")
    assert feedback.approved is True


def test_more_cautious_than_ground_truth_is_still_approved():
    """Asking when silence would've sufficed is never punished."""
    email = make_email(ground_truth_decision=Decision.PROCEED_SILENT)
    outcome = make_outcome(Decision.ASK_FIRST)
    feedback = RuleOracle().give_feedback(email, outcome, bucket="b")
    assert feedback.approved is True


def test_more_autonomous_than_ground_truth_is_rejected():
    """The dangerous direction: overconfidence relative to the label."""
    email = make_email(ground_truth_decision=Decision.ASK_FIRST)
    outcome = make_outcome(Decision.PROCEED_SILENT)
    feedback = RuleOracle().give_feedback(email, outcome, bucket="b")
    assert feedback.approved is False
    assert "too autonomous" in feedback.note


# ---------------------------------------------------------------------------
# NoisyRuleOracle
# ---------------------------------------------------------------------------

def test_zero_noise_matches_rule_oracle_exactly():
    email = make_email(ground_truth_decision=Decision.PROCEED_SILENT)
    outcome = make_outcome(Decision.ASK_FIRST)
    feedback = NoisyRuleOracle(noise=0.0).give_feedback(email, outcome, bucket="b")
    assert feedback.approved is True
    assert feedback.corrected is False


def test_full_noise_always_flips_rule_oracle_baseline():
    email = make_email(ground_truth_decision=Decision.ASK_FIRST)
    outcome = make_outcome(Decision.PROCEED_SILENT)  # baseline: rejected
    feedback = NoisyRuleOracle(noise=1.0).give_feedback(email, outcome, bucket="b")
    assert feedback.approved is True  # flipped
    assert feedback.corrected is True


def test_noisy_oracle_is_deterministic_for_a_fixed_seed():
    email = make_email(ground_truth_decision=Decision.PROCEED_NOTIFY)
    outcome = make_outcome(Decision.PROCEED_NOTIFY)
    first = NoisyRuleOracle(noise=0.5, seed=42).give_feedback(email, outcome, bucket="b")
    second = NoisyRuleOracle(noise=0.5, seed=42).give_feedback(email, outcome, bucket="b")
    assert first.approved == second.approved
    assert first.corrected == second.corrected


# ---------------------------------------------------------------------------
# LLMPersonaOracle: success, malformed response, and infra-failure fallback
# ---------------------------------------------------------------------------

def test_llm_persona_uses_llm_result_when_call_succeeds():
    def fake_call(system, user):
        return '{"approved": true, "corrected": false, "note": "seems fine"}'

    oracle = LLMPersonaOracle(call_fn=fake_call)
    email = make_email()
    outcome = make_outcome(Decision.PROCEED_NOTIFY)

    feedback = oracle.give_feedback(email, outcome, bucket="b")

    assert oracle.last_source_used == "llm_persona"
    assert feedback.approved is True
    assert feedback.note == "seems fine"


def test_llm_persona_strips_markdown_code_fences_before_parsing():
    def fenced_call(system, user):
        return '```json\n{"approved": false, "corrected": true, "note": "too risky"}\n```'

    oracle = LLMPersonaOracle(call_fn=fenced_call)
    feedback = oracle.give_feedback(make_email(), make_outcome(Decision.PROCEED_SILENT), bucket="b")

    assert oracle.last_source_used == "llm_persona"
    assert feedback.approved is False
    assert feedback.corrected is True


def test_llm_persona_falls_back_when_call_raises():
    def failing_call(system, user):
        raise RuntimeError("simulated network failure")

    fallback = RuleOracle()
    oracle = LLMPersonaOracle(call_fn=failing_call, fallback=fallback)
    email = make_email(ground_truth_decision=Decision.PROCEED_SILENT)
    outcome = make_outcome(Decision.PROCEED_SILENT)

    feedback = oracle.give_feedback(email, outcome, bucket="b")

    assert oracle.last_source_used == "fallback"
    assert feedback.approved is True  # delegated to RuleOracle, which approves a match


def test_llm_persona_falls_back_when_response_is_malformed_json():
    def bad_call(system, user):
        return "not json at all"

    fallback = RuleOracle()
    oracle = LLMPersonaOracle(call_fn=bad_call, fallback=fallback)
    email = make_email(ground_truth_decision=None)

    feedback = oracle.give_feedback(email, make_outcome(Decision.PROCEED_SILENT), bucket="b")

    assert oracle.last_source_used == "fallback"
    assert feedback.approved is True  # RuleOracle's no-label neutral approval


def test_llm_persona_falls_back_when_response_missing_required_field():
    def missing_field_call(system, user):
        return '{"corrected": false, "note": "forgot approved"}'

    oracle = LLMPersonaOracle(call_fn=missing_field_call, fallback=RuleOracle())
    feedback = oracle.give_feedback(make_email(), make_outcome(Decision.PROCEED_SILENT), bucket="b")

    assert oracle.last_source_used == "fallback"
