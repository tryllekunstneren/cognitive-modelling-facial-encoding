"""
Analyse Experiment 1 data: one histogram per participant, check whether
each participant used the full 1-5 rating range, and min-max normalise
any participant who did not (so all participants' ratings sit on the
same 1-5 scale before they get pooled for the encoding model).

Reads every data/experiment1/<student_id>.csv (skips *_trials.csv), where
each file has: filename, rating_1, rating_2 (no header).

Writes:
  data/experiment1/plots/hist_<student_id>.png   - one histogram per participant
  data/experiment1/<student_id>_normalised.csv   - only written if that
                                                    participant's raw range
                                                    was not 1-5
  Prints a summary table of each participant's observed min/max and
  whether normalisation was applied.

Run:  python scripts/6_analyze_experiment1.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
EXP_DIR = BASE / "data" / "experiment1"
PLOTS_DIR = EXP_DIR / "plots"

SCALE_MIN, SCALE_MAX = 1, 5


def min_max_to_scale(values: pd.Series, lo: float, hi: float) -> pd.Series:
    """Rescale values from [lo, hi] to [SCALE_MIN, SCALE_MAX]."""
    if hi == lo:
        return pd.Series([np.nan] * len(values), index=values.index)
    return SCALE_MIN + (values - lo) * (SCALE_MAX - SCALE_MIN) / (hi - lo)


def main():
    participant_files = sorted(
        p for p in EXP_DIR.glob("*.csv") if not p.name.endswith("_trials.csv")
        and "_normalised" not in p.name
    )
    if not participant_files:
        raise SystemExit(f"No participant CSVs found in {EXP_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for path in participant_files:
        student_id = path.stem
        df = pd.read_csv(path, header=None, names=["filename", "rating_1", "rating_2"])
        all_ratings = pd.concat([df.rating_1, df.rating_2]).dropna()
        all_ratings = pd.to_numeric(all_ratings, errors="coerce").dropna()

        obs_min, obs_max = all_ratings.min(), all_ratings.max()
        full_range = (obs_min == SCALE_MIN) and (obs_max == SCALE_MAX)

        # histogram
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(all_ratings, bins=np.arange(SCALE_MIN, SCALE_MAX + 2) - 0.5,
                rwidth=0.8, color="#4C72B0", edgecolor="white")
        ax.set_xticks(range(SCALE_MIN, SCALE_MAX + 1))
        ax.set_xlabel("Rating (1 = very sad, 5 = very happy)")
        ax.set_ylabel("Count")
        ax.set_title(f"Participant {student_id}  (n={len(all_ratings)})")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"hist_{student_id}.png", dpi=150)
        plt.close(fig)

        normalised_written = False
        if not full_range:
            df_norm = df.copy()
            df_norm["rating_1"] = min_max_to_scale(
                pd.to_numeric(df_norm.rating_1, errors="coerce"), obs_min, obs_max)
            df_norm["rating_2"] = min_max_to_scale(
                pd.to_numeric(df_norm.rating_2, errors="coerce"), obs_min, obs_max)
            df_norm.to_csv(EXP_DIR / f"{student_id}_normalised.csv",
                            index=False, header=False)
            normalised_written = True

        summary_rows.append({
            "student_id": student_id,
            "n_ratings": len(all_ratings),
            "observed_min": obs_min,
            "observed_max": obs_max,
            "used_full_1_5_range": full_range,
            "normalised_file_written": normalised_written,
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    print(f"\nHistograms saved to {PLOTS_DIR}")
    if (~summary.used_full_1_5_range).any():
        print("\nSome participants did not use the full 1-5 range; "
              "min-max normalised versions were written next to their "
              "original files (suffix _normalised.csv).")
    else:
        print("\nAll participants used the full 1-5 range; no "
              "normalisation was necessary.")


if __name__ == "__main__":
    main()
