"""
Shared loader for the sample-inbox JSON format, used by both `main.py` and
`eval/run_eval.py` (which previously duplicated this exact parsing logic).

Validates the fields the rest of the pipeline assumes are present, so a
malformed or incomplete record in `data/inbox_sample.json` raises a clear
`ValueError` naming the offending record and field, instead of an opaque
`KeyError` deep inside pipeline code.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.schemas import Decision, Email

REQUIRED_FIELDS = ("id", "sender", "sender_domain", "subject", "body", "known_sender")


def _email_from_record(record: dict) -> Email:
    record_id = record.get("id", "<missing id>")
    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            raise ValueError(
                f"email record {record_id!r} is missing required field '{field_name}'"
            )

    raw_gt_decision = record.get("ground_truth_decision")
    try:
        ground_truth_decision = Decision(raw_gt_decision) if raw_gt_decision else None
    except ValueError as e:
        raise ValueError(
            f"email record {record_id!r} has invalid ground_truth_decision "
            f"{raw_gt_decision!r}"
        ) from e

    return Email(
        id=record["id"],
        sender=record["sender"],
        sender_domain=record["sender_domain"],
        subject=record["subject"],
        body=record["body"],
        known_sender=record["known_sender"],
        thread_has_prior_context=record.get("thread_has_prior_context", False),
        ground_truth_category=record.get("ground_truth_category"),
        ground_truth_decision=ground_truth_decision,
    )


def load_emails_from_json(path: Path) -> list[Email]:
    raw = json.loads(Path(path).read_text())
    return [_email_from_record(r) for r in raw]
