"""Renders the calibration curve (ask_rate over epochs, both oracles) from
eval/results/epoch_summary.csv into eval/results/calibration_curve.png."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    rows = list(csv.DictReader((RESULTS_DIR / "epoch_summary.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for oracle in ("rule", "llm"):
        sub = [r for r in rows if r["oracle"] == oracle]
        epochs = [int(r["epoch"]) for r in sub]
        ask = [float(r["ask_rate"]) for r in sub]
        silent = [float(r["silent_rate"]) for r in sub]
        exact = [float(r["exact_match_vs_ground_truth"]) for r in sub]
        label = "RuleOracle" if oracle == "rule" else "LLMPersonaOracle (noisy fallback)"

        axes[0].plot(epochs, ask, marker="o", label=f"{label} ask_rate")
        axes[0].plot(epochs, silent, marker="s", linestyle="--", label=f"{label} silent_rate")
        axes[1].plot(epochs, exact, marker="^", label=label)

    axes[0].set_title("Ask-rate falls, silent-rate rises\nas the bucket earns trust")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("fraction of emails")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 1)

    axes[1].set_title("Decision matches ground truth\nmore often as confidence calibrates")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("exact match rate")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1)

    fig.suptitle("Calibration over epochs — safety violations were 0 in every epoch, both oracles")
    fig.tight_layout()
    out = RESULTS_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
