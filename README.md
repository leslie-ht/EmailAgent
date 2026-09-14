# Proactive Email Agent

A take-home project: an email agent that decides, per incoming message, how
much autonomy to take — `proceed_silent`, `proceed_notify`, `ask_first`, or
`escalate` — and calibrates over time from user feedback, without ever
letting learning weaken a hard safety floor (irreversible / external /
money / prompt-injection).

See **[DESIGN.md](DESIGN.md)** for the full design writeup and eval
results, and **[transcripts/examples.md](transcripts/examples.md)** for
annotated example runs.

## Setup

```bash
pip install -r requirements.txt   # only needed for tests / plotting / real LLM calls
export ANTHROPIC_API_KEY=sk-...   # optional — omit to run fully offline on heuristics
```

The agent runs with **zero dependencies** if you skip the LLM path (the
`HybridClassifier` falls back to deterministic heuristics automatically).
`pytest` and `matplotlib` are only needed for running tests and regenerating
the calibration chart, respectively.

## Usage

```bash
# Narrated walkthrough: cold start -> earned autonomy -> floor holding -> injection escalate
python main.py demo

# Single pass over the whole sample inbox (no learning loop)
python main.py batch

# Multi-epoch calibration study (the eval harness) — this is what generates
# the numbers and chart referenced in DESIGN.md
python eval/run_eval.py --epochs 10 --oracle both
python eval/plot_calibration.py   # regenerates eval/results/calibration_curve.png

# Regenerate the labeled synthetic dataset (deterministic, seeded)
python data/generate_inbox.py

# Run the test suite (safety gate + policy)
pytest tests/ -v
```

## Project layout

```
agent/
  schemas.py       # Email, Decision, Action, Feedback, etc.
  classifier.py     # Hybrid LLM + heuristic classifier
  safety_gate.py     # THE hard floor — deterministic, no learned state
  policy.py            # Beta-Bernoulli bandit + 4-way decision combining
  executor.py            # Mocked action execution (no real sends)
  oracle.py                # RuleOracle + LLMPersonaOracle feedback simulators
data/
  generate_inbox.py         # Synthetic labeled dataset generator
  inbox_sample.json           # 77 labeled emails across 11 categories
eval/
  run_eval.py                  # Multi-epoch calibration harness
  metrics.py                     # Independent safety-violation / accuracy checks
  plot_calibration.py              # Chart generation
  results/                           # CSV + PNG output (committed as evidence)
transcripts/
  examples.md                        # Curated annotated transcripts
tests/
  test_safety_gate.py                  # Exhaustive floor tests
  test_policy.py                         # Bandit + floor-interaction tests
main.py                                    # CLI entrypoint
DESIGN.md                                    # Full design writeup
```

## The one thing to look at first

If you only run one thing: `python main.py demo`. It walks through cold
start → earned autonomy on a safe bucket → the safety floor holding firm on
a risky bucket *despite identical positive feedback* → an injection attempt
escalating with zero prior history. That sequence is the whole point of the
project.
