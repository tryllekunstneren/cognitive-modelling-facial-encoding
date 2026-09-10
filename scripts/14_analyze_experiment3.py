"""
Analyse Experiment 3 data: perceptual after-effect (exercise step 8).

For each participant, and pooled across all participants, plots mean
rating for each of the 3 test stimuli, separately for the two adaptation
conditions (sad_endpoint vs. happy_endpoint).

The predicted after-effect is contrastive: adapting to the SAD endpoint
should make a neutral test face look relatively HAPPIER (higher rating),
and adapting to the HAPPY endpoint should make it look relatively SADDER
(lower rating). So we expect mean_rating(sad_endpoint) >
mean_rating(happy_endpoint) for every test stimulus.

Run:  python scripts/14_analyze_experiment3.py
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
EXP_DIR = BASE / "data" / "experiment3"
PLOTS_DIR = EXP_DIR / "plots"


def plot_after_effect(df, title, out_path):
    grouped = df.groupby(["test_target_rating", "adapt_label"])["response"].agg(["mean", "sem"]).reset_index()
    test_levels = sorted(grouped.test_target_rating.unique())

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, marker, color in [("sad_endpoint", "o", "#4C72B0"), ("happy_endpoint", "s", "#C44E52")]:
        sub = grouped[grouped.adapt_label == label].sort_values("test_target_rating")
        ax.errorbar(sub.test_target_rating, sub["mean"], yerr=sub["sem"].fillna(0),
                    marker=marker, label=f"adapted to {label}", color=color, capsize=4)
    ax.set_xticks(test_levels)
    ax.set_xlabel("Test stimulus (target rating it was generated at)")
    ax.set_ylabel("Mean rating given by participant")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return grouped


def summarise(grouped, label):
    pivot = grouped.pivot(index="test_target_rating", columns="adapt_label", values="mean")
    pivot["diff_sad_minus_happy"] = pivot["sad_endpoint"] - pivot["happy_endpoint"]
    pivot["as_predicted"] = pivot["diff_sad_minus_happy"] > 0
    print(f"\n{label}:")
    print(pivot.to_string())
    return pivot


def main():
    trial_files = sorted(EXP_DIR.glob("*_trials.csv"))
    if not trial_files:
        raise SystemExit(f"No Experiment 3 data found in {EXP_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for path in trial_files:
        student_id = path.stem.replace("_trials", "")
        df = pd.read_csv(path)
        df["student_id"] = student_id
        all_dfs.append(df)

        grouped = plot_after_effect(
            df, f"Experiment 3 after-effect - participant {student_id}",
            PLOTS_DIR / f"aftereffect_{student_id}.png")
        summarise(grouped, f"Participant {student_id}")

    combined = pd.concat(all_dfs, ignore_index=True)
    grouped_all = plot_after_effect(
        combined, "Experiment 3 after-effect - all participants pooled",
        PLOTS_DIR / "aftereffect_pooled.png")
    pivot_all = summarise(grouped_all, "Pooled (all participants)")

    n_predicted = pivot_all["as_predicted"].sum()
    n_total = len(pivot_all)
    print(f"\n{n_predicted}/{n_total} test stimuli show the predicted direction "
          f"(rating after sad-endpoint adaptation > rating after happy-endpoint "
          f"adaptation) in the pooled data.")
    print(f"\nPlots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
