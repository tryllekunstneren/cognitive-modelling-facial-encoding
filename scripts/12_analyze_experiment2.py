"""
Analyse Experiment 2 data (exercise step 7).

For each participant's data/experiment2/<student_id>_trials.csv:
  - box plot of actual ratings, grouped by the synthetic image's target
    (model-predicted) rating
  - Spearman's rank correlation, rho, between target rating and actual
    rating (computed both trial-by-trial and on each image's mean rating)

Run:  python scripts/12_analyze_experiment2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
EXP_DIR = BASE / "data" / "experiment2"
PLOTS_DIR = EXP_DIR / "plots"


def main():
    trial_files = sorted(EXP_DIR.glob("*_trials.csv"))
    if not trial_files:
        raise SystemExit(f"No Experiment 2 data found in {EXP_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for path in trial_files:
        student_id = path.stem.replace("_trials", "")
        df = pd.read_csv(path)

        rho_trials, p_trials = spearmanr(df["target_rating"], df["rating"])

        per_image = df.groupby("target_rating")["rating"].mean().reset_index()
        rho_means, p_means = spearmanr(per_image["target_rating"], per_image["rating"])

        levels = sorted(df["target_rating"].unique())
        grouped = [df.loc[df.target_rating == lvl, "rating"].values for lvl in levels]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(grouped, positions=range(len(levels)), widths=0.6)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{lvl:.2f}" for lvl in levels], rotation=45)
        ax.set_xlabel("Model-predicted (target) rating of synthetic image")
        ax.set_ylabel("Actual rating given by participant (1-5)")
        ax.set_title(f"Experiment 2 - participant {student_id}\n"
                      f"Spearman rho (trial-level) = {rho_trials:.3f}, "
                      f"rho (image-mean) = {rho_means:.3f}")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"boxplot_{student_id}.png", dpi=150)
        plt.close(fig)

        summary_rows.append({
            "student_id": student_id,
            "n_trials": len(df),
            "n_images": len(levels),
            "spearman_rho_trials": rho_trials,
            "spearman_p_trials": p_trials,
            "spearman_rho_image_means": rho_means,
            "spearman_p_image_means": p_means,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(EXP_DIR / "experiment2_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nBox plots saved to {PLOTS_DIR}")
    print(f"Summary saved to {EXP_DIR / 'experiment2_summary.csv'}")

    for _, row in summary.iterrows():
        closeness = "close to 1 (strong validation)" if row.spearman_rho_trials > 0.7 \
            else "moderate" if row.spearman_rho_trials > 0.4 else "weak"
        print(f"{row.student_id}: rho={row.spearman_rho_trials:.3f} -> {closeness}")


if __name__ == "__main__":
    main()
