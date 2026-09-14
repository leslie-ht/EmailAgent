"""
Multi-epoch eval harness.

Replays the labeled synthetic inbox across N epochs (shuffled each time),
feeding the agent's decisions back through a feedback oracle so the
bandit-based policy calibrates over time. Reports, per epoch, per oracle:

  - ask_rate / escalate_rate / notify_rate / silent_rate
  - exact_match_vs_ground_truth
  - safety_violations (must be 0, always -- checked independently of the
    policy module's own floor-enforcement, as a second line of defense
    in the eval itself)
  - overautonomy_vs_ground_truth (decision more autonomous than the label)
  - classifier_fallback_rate (how often the LLM path failed and heuristic
    kicked in -- always 100% in this offline sandbox since no API key is
    configured, which is itself a useful reliability data point)

Run under BOTH RuleOracle and LLMPersonaOracle (falls back to RuleOracle
internally if no LLM available) to compare calibration behavior under a
noisier feedback signal.

Usage: python eval/run_eval.py [--epochs 8] [--oracle rule|llm|both]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from agent.classifier import HybridClassifier
from agent.executor import execute
from agent.oracle import BaseOracle, LLMPersonaOracle, RuleOracle
from agent.policy import PolicyState, decide
from agent.schemas import Decision, Email
from eval.metrics import is_overautonomy_vs_ground_truth, is_safety_violation, summarize_epoch

DATA_PATH = Path(__file__).parent.parent / "data" / "inbox_sample.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_emails() -> list[Email]:
    raw = json.loads(DATA_PATH.read_text())
    emails = []
    for r in raw:
        gt = Decision(r["ground_truth_decision"]) if r.get("ground_truth_decision") else None
        emails.append(Email(
            id=r["id"], sender=r["sender"], sender_domain=r["sender_domain"],
            subject=r["subject"], body=r["body"], known_sender=r["known_sender"],
            thread_has_prior_context=r.get("thread_has_prior_context", False),
            ground_truth_category=r.get("ground_truth_category"),
            ground_truth_decision=gt,
        ))
    return emails


def run(oracle_name: str, epochs: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    emails = load_emails()
    classifier = HybridClassifier()
    state = PolicyState()
    oracle: BaseOracle = RuleOracle() if oracle_name == "rule" else LLMPersonaOracle()

    epoch_summaries = []
    for epoch in range(1, epochs + 1):
        order = emails[:]
        rng.shuffle(order)
        records = []
        for email in order:
            classification = classifier.classify(email)
            result = decide(email, classification, state)
            outcome = execute(email, result)

            feedback = oracle.give_feedback(email, outcome, result.bucket)
            state.record_feedback(feedback)

            gt = email.ground_truth_decision
            records.append({
                "email_id": email.id,
                "category": email.ground_truth_category,
                "decision": result.decision.value,
                "ground_truth_decision": gt.value if gt else None,
                "safety_min_decision": result.safety.min_decision.value,
                "safety_violation": is_safety_violation(email, result.decision, result.safety.min_decision),
                "overautonomy_vs_gt": is_overautonomy_vs_ground_truth(result.decision, gt),
                "classifier_source": classification.source if classification.source == "llm" else classifier.last_source_used,
                "confidence_lower_bound": result.confidence_lower_bound,
            })

        summary = summarize_epoch(records)
        summary["epoch"] = epoch
        summary["oracle"] = oracle_name
        epoch_summaries.append(summary)
        print(f"[{oracle_name}] epoch {epoch}: ask_rate={summary['ask_rate']:.2f} "
              f"silent_rate={summary['silent_rate']:.2f} exact_match={summary['exact_match_vs_ground_truth']:.2f} "
              f"safety_violations={summary['safety_violations']} overautonomy={summary['overautonomy_vs_ground_truth']}")

    return epoch_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--oracle", choices=["rule", "llm", "both"], default="both")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    all_summaries = []

    oracles = ["rule", "llm"] if args.oracle == "both" else [args.oracle]
    for oracle_name in oracles:
        all_summaries.extend(run(oracle_name, args.epochs))

    out_path = RESULTS_DIR / "epoch_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        for row in all_summaries:
            writer.writerow(row)
    print(f"\nWrote {len(all_summaries)} epoch summaries to {out_path}")

    total_violations = sum(s["safety_violations"] for s in all_summaries)
    print(f"\nTOTAL SAFETY VIOLATIONS ACROSS ALL EPOCHS/ORACLES: {total_violations}")


if __name__ == "__main__":
    main()
