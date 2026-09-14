"""
CLI entrypoint.

  python main.py demo                  -> run a handful of illustrative
                                             emails through the full
                                             pipeline with verbose trace
  python main.py batch                 -> run the whole sample inbox once
                                             (single pass, no learning
                                             loop -- see eval/run_eval.py
                                             for the calibration study)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.classifier import HybridClassifier
from agent.executor import execute
from agent.policy import PolicyState, decide
from agent.schemas import Decision, Email

DATA_PATH = Path(__file__).parent / "data" / "inbox_sample.json"


def load_emails() -> list[Email]:
    raw = json.loads(DATA_PATH.read_text())
    emails = []
    for r in raw:
        gt = Decision(r["ground_truth_decision"]) if r.get("ground_truth_decision") else None
        emails.append(Email(
            id=r["id"], sender=r["sender"], sender_domain=r["sender_domain"],
            subject=r["subject"], body=r["body"], known_sender=r["known_sender"],
            thread_has_prior_context=r.get("thread_has_prior_context", False),
            ground_truth_category=r.get("ground_truth_category"), ground_truth_decision=gt,
        ))
    return emails


def trace_email(email: Email, classifier: HybridClassifier, state: PolicyState, verbose: bool = True):
    classification = classifier.classify(email)
    result = decide(email, classification, state)
    outcome = execute(email, result)

    if verbose:
        print(f"\n--- Email {email.id}: {email.subject!r} (from {email.sender}) ---")
        print(f"  Classifier source : {classifier.last_source_used or classification.source}")
        print(f"  Category          : {classification.category}")
        print(f"  Proposed action   : {classification.proposed_action.value}")
        print(f"  Stakes tier       : {classification.stakes_tier.value}")
        if result.safety.is_gated:
            print(f"  Safety gate       : GATED -> {result.safety.reasons}")
        else:
            print(f"  Safety gate       : clear")
        print(f"  Learned confidence (lower bound): {result.confidence_lower_bound:.2f}  [bucket: {result.bucket}]")
        print(f"  FINAL DECISION    : {result.decision.value}")
        print(f"  {outcome.log_message}")
    return result, outcome


def demo():
    """
    Narrated walkthrough designed to show the three behaviors that matter
    most: (1) cautious cold start, (2) autonomy increasing after positive
    feedback for a safe bucket, (3) the safety floor holding firm for a
    risky bucket EVEN AFTER the same amount of positive feedback, and
    (4) an injection attempt escalating immediately regardless of history.
    """
    from agent.schemas import Feedback

    emails = load_emails()
    by_category = {}
    for e in emails:
        by_category.setdefault(e.ground_truth_category, []).append(e)

    classifier = HybridClassifier()
    state = PolicyState()

    print("=" * 70)
    print("STEP 1: cold start on a newsletter email (no history yet)")
    print("=" * 70)
    newsletter_emails = by_category["newsletter"]
    result, outcome = trace_email(newsletter_emails[0], classifier, state)

    print("\n" + "=" * 70)
    print("STEP 2: simulate 25 rounds of positive feedback on newsletter::archive")
    print("=" * 70)
    for e in (newsletter_emails * 5)[:25]:
        r, o = trace_email(e, classifier, state, verbose=False)
        state.record_feedback(Feedback(email_id=e.id, bucket=r.bucket, approved=True))
    print("...(25 approvals recorded, log suppressed)...")
    trace_email(newsletter_emails[-1], classifier, state)
    print(">>> Autonomy increased to proceed_silent for this now-trusted bucket.\n")

    print("=" * 70)
    print("STEP 3: same 25-round positive feedback, but on spam::delete (irreversible)")
    print("=" * 70)
    spam_emails = by_category["spam"]
    for e in (spam_emails * 5)[:25]:
        r, o = trace_email(e, classifier, state, verbose=False)
        state.record_feedback(Feedback(email_id=e.id, bucket=r.bucket, approved=True))
    print("...(25 approvals recorded, log suppressed)...")
    trace_email(spam_emails[-1], classifier, state)
    print(">>> Despite identical positive feedback, decision stayed at ask_first:")
    print(">>> delete is irreversible, so the safety gate's floor caps it regardless of learned confidence.\n")

    print("=" * 70)
    print("STEP 4: a prompt-injection attempt, even with zero prior history")
    print("=" * 70)
    injection_emails = by_category["phishing_or_injection"]
    trace_email(injection_emails[0], classifier, state)
    print(">>> Escalates immediately -- injection detection does not depend on learned state at all.\n")


def batch():
    emails = load_emails()
    classifier = HybridClassifier()
    state = PolicyState()
    for email in emails:
        trace_email(email, classifier, state, verbose=False)
    print(f"Processed {len(emails)} emails.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["demo", "batch"])
    args = parser.parse_args()
    if args.mode == "demo":
        demo()
    else:
        batch()
