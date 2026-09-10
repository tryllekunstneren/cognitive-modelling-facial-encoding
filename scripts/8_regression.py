"""
Linear encoding model with forward-selected PCs (exercise step 4).

- Loads the PCA scores + mean ratings from data/pca/scores.csv (created by
  scripts/7_pca.py - re-run that first if this file is missing, it's
  gitignored since it's fully regenerable).
- Restricts candidate predictors to the top N PCs that explain 95% of
  variance (data/pca/n_pcs_recommended.txt), matching the dimensionality
  reduction done in step 3.
- Uses sklearn's SequentialFeatureSelector (forward, 5-fold CV, R^2) to
  pick the relevant PCs, then fits the final LinearRegression on just
  those PCs.
- Because PCA components are orthonormal, the fitted coefficient vector
  doubles as the model's weight vector w in pixel space once mapped back
  through the selected components - this is what scripts/9 (synthetic
  image generation) uses.
- Visualises the selected PCs as min/average/max triplets, same style as
  step 3, so you can compare "PCs that explain image variance" against
  "PCs that actually predict happy/sad".

Saves to data/regression/:
  selected_pcs.txt       - which PC indices (1-based) were selected
  model_coef.npy         - regression coefficients, one per selected PC
  model_intercept.txt    - regression intercept
  cv_r2.txt              - mean cross-validated R^2 of the final model
  plots/selected_pc_triplets.png

Run:  python scripts/8_regression.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PCA_DIR = BASE / "data" / "pca"
REG_DIR = BASE / "data" / "regression"
PLOTS_DIR = REG_DIR / "plots"

N_SPLITS = 5
RANDOM_SEED = 42


def main():
    scores_path = PCA_DIR / "scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found. Run scripts/7_pca.py first.")

    df = pd.read_csv(scores_path)
    n_keep = int((PCA_DIR / "n_pcs_recommended.txt").read_text().strip())
    pc_cols = [f"PC{i+1}" for i in range(n_keep)]

    X = df[pc_cols].values
    y = df["mean_rating"].values
    print(f"{X.shape[0]} images, {X.shape[1]} candidate PCs (top {n_keep}, 95% variance), "
          f"rating range [{y.min():.2f}, {y.max():.2f}]")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    baseline_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2").mean()
    print(f"Baseline CV R^2 using all {n_keep} candidate PCs (likely overfit): {baseline_r2:.3f}")

    selector = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=0.005,
        direction="forward",
        scoring="r2",
        cv=cv,
    )
    selector.fit(X, y)
    selected_mask = selector.get_support()
    selected_idx = np.where(selected_mask)[0]  # 0-based index into pc_cols
    selected_pcs = [i + 1 for i in selected_idx]  # 1-based PC numbers
    print(f"\nForward selection picked {len(selected_pcs)} PCs: {selected_pcs}")

    X_sel = X[:, selected_idx]
    cv_r2 = cross_val_score(LinearRegression(), X_sel, y, cv=cv, scoring="r2").mean()
    print(f"Cross-validated R^2 with selected PCs: {cv_r2:.3f}")

    model = LinearRegression().fit(X_sel, y)
    print(f"Intercept: {model.intercept_:.3f}")
    print("Coefficients:")
    for pc, coef in zip(selected_pcs, model.coef_):
        print(f"  PC{pc}: {coef:+.4f}")

    w_norm = np.linalg.norm(model.coef_)
    print(f"\n||w|| (norm of weight vector in PCA-score space) = {w_norm:.4f}")

    REG_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REG_DIR / "model_coef.npy", model.coef_)
    np.save(REG_DIR / "selected_pc_indices.npy", np.array(selected_idx))  # 0-based
    (REG_DIR / "selected_pcs.txt").write_text(",".join(str(p) for p in selected_pcs))
    (REG_DIR / "model_intercept.txt").write_text(str(model.intercept_))
    (REG_DIR / "cv_r2.txt").write_text(str(cv_r2))
    print(f"\nSaved model to {REG_DIR}")

    # --- visualise selected PCs as min/average/max triplets ---
    mean_image = np.load(PCA_DIR / "mean_image.npy")
    components = np.load(PCA_DIR / "components.npy")
    shape = tuple(int(x) for x in (PCA_DIR / "image_shape.txt").read_text().split(","))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n_show = len(selected_pcs)
    fig, axes = plt.subplots(n_show, 3, figsize=(6, 2 * n_show), squeeze=False)
    for row, (pc_num, idx0) in enumerate(zip(selected_pcs, selected_idx)):
        pc_vector = components[idx0]
        pc_scores = X[:, idx0]
        lo_img = (mean_image + pc_scores.min() * pc_vector).reshape(shape)
        hi_img = (mean_image + pc_scores.max() * pc_vector).reshape(shape)
        mid_img = mean_image.reshape(shape)
        for ax, img, title in zip(axes[row], [lo_img, mid_img, hi_img],
                                   [f"PC{pc_num} min", "average", f"PC{pc_num} max"]):
            ax.imshow(np.clip(img, 0, 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "selected_pc_triplets.png", dpi=150)
    plt.close(fig)
    print(f"Saved selected-PC visualisation to {PLOTS_DIR / 'selected_pc_triplets.png'}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Report answer - "In your report, visualise some of the principal
# components that were selected in the same way as you did in step 3.
# Are the features relevant to the rating task?"
#
# See data/regression/plots/selected_pc_triplets.png (min/average/max
# triplets for the 17 forward-selected PCs: 1, 3, 5, 8, 9, 11, 13, 14, 15,
# 16, 18, 19, 22, 27, 43, 52, 73).
#
# A subset of the selected PCs show a visible change in mouth shape between
# their min and max faces - most clearly PC5, PC9, PC14 and PC22, where the
# max face has a visibly more open/upturned mouth than the min face. These
# look plausibly related to the happy/sad rating task.
#
# Several other selected PCs (PC1, PC3, PC8, PC13, PC16) mainly vary in
# overall brightness/contrast or head angle rather than expression - the
# same nuisance lighting/pose variation already flagged for PC1 in step 3.
# Forward selection can still pick these up if they happen to correlate
# with the ratings in this particular sample of 200 images (e.g. if
# lighting condition happened to coincide with expression during
# selection), without representing a genuine, generalisable encoding of
# happy/sad.
#
# The higher-index components (PC43, PC52, PC73) explain very little
# overall image variance and their min/max faces show only subtle,
# localised texture differences that are hard to interpret as expression
# at all - these more likely reflect fine per-image idiosyncrasies
# (individual identity detail) than a shared expression dimension.
#
# Overall: the model is picking up a real, interpretable expression signal
# (via PC5/PC9/PC14/PC22 in particular), but it is mixed in with nuisance
# and possibly-overfit components, consistent with the moderate
# cross-validated R^2 (~0.58) rather than a very high one. This is worth
# discussing as a limitation in the report: with only 200 images and 2
# raters, forward selection has enough freedom to include a few PCs that
# fit this specific sample without necessarily capturing a robust,
# generalisable "happy/sad" direction.
# ---------------------------------------------------------------------------
