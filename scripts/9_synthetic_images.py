"""
Generate synthetic images from the fitted encoding model (exercise step 5).

The model predicts:  rating = intercept + w . scores
where w (in PCA-score space) is the fitted regression coefficient vector
from scripts/8_regression.py, over the selected PCs only.

Because PCA components are orthonormal, mapping w back into pixel space
via w_pixel = sum(coef_i * component_i) gives a vector with the same norm,
so the minimum-norm image that reaches a target rating is:

    alpha = (target_rating - intercept) / ||w_pixel||^2
    image = mean_image + alpha * w_pixel

This is Eq. 2.22/2.27 in Herlau et al. (the reading for this exercise):
the pseudo-inverse solution that moves only along the model's weight
direction (the "shortest path" in pixel space to reach a given rating),
leaving every other direction untouched.

Generates 11 faces: ratings 1-5 plus the six half-points 0.5, 1.5, 2.5,
3.5, 4.5, 5.5, and also reports the predicted rating range on the training
images themselves (needed for step 6, to check whether the synthetic faces
extrapolate beyond the training distribution).

Run:  python scripts/9_synthetic_images.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PCA_DIR = BASE / "data" / "pca"
REG_DIR = BASE / "data" / "regression"
SYN_DIR = BASE / "data" / "synthetic"
PLOTS_DIR = SYN_DIR / "plots"

TARGET_RATINGS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 0.5, 5.5]
# Displayed sorted, but generated in this order so the 5 "on-scale" faces
# come first, matching the 5 rating levels used in Experiment 1.


def main():
    mean_image = np.load(PCA_DIR / "mean_image.npy")
    components = np.load(PCA_DIR / "components.npy")
    shape = tuple(int(x) for x in (PCA_DIR / "image_shape.txt").read_text().split(","))

    selected_idx = np.load(REG_DIR / "selected_pc_indices.npy")  # 0-based
    coef = np.load(REG_DIR / "model_coef.npy")
    intercept = float((REG_DIR / "model_intercept.txt").read_text())

    # weight vector in pixel space (valid because PCA components are orthonormal)
    w_pixel = coef @ components[selected_idx]
    w_normsq = float(np.dot(coef, coef))
    print(f"||w||^2 = {w_normsq:.6e}  (small -> large alpha -> risk of extrapolating "
          f"beyond the training data, see step 6)")

    # sanity check: predicted rating range on the training images themselves
    scores_df = pd.read_csv(PCA_DIR / "scores.csv")
    pc_cols = [f"PC{i+1}" for i in selected_idx]
    X_sel = scores_df[pc_cols].values
    predicted = intercept + X_sel @ coef
    print(f"Predicted rating range on the 200 training images: "
          f"[{predicted.min():.2f}, {predicted.max():.2f}]  "
          f"(actual rating range: [{scores_df.mean_rating.min():.2f}, "
          f"{scores_df.mean_rating.max():.2f}])")

    SYN_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    images_by_rating = {}
    for target in TARGET_RATINGS:
        alpha = (target - intercept) / w_normsq
        img = (mean_image + alpha * w_pixel).reshape(shape)
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        images_by_rating[target] = img_clipped

        out_name = f"synthetic_rating_{target:.1f}.png"
        Image.fromarray(img_clipped).save(SYN_DIR / out_name)
        manifest_rows.append({
            "target_rating": target,
            "alpha": alpha,
            "filename": out_name,
            "pixel_min_before_clip": float(img.min()),
            "pixel_max_before_clip": float(img.max()),
        })
        print(f"target={target:.1f}  alpha={alpha:+.2f}  "
              f"pixel range before clipping=[{img.min():.0f}, {img.max():.0f}]")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SYN_DIR / "synthetic_manifest.csv", index=False)

    # composite figure, sorted by rating, matching Figures 2.6/2.7 style
    sorted_ratings = sorted(images_by_rating)
    fig, axes = plt.subplots(1, len(sorted_ratings), figsize=(2 * len(sorted_ratings), 2.4))
    for ax, target in zip(axes, sorted_ratings):
        ax.imshow(images_by_rating[target], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{target:.1f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Synthetic faces across the happy/sad rating continuum")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "synthetic_faces_row.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 11 synthetic images + manifest to {SYN_DIR}")
    print(f"Saved composite figure to {PLOTS_DIR / 'synthetic_faces_row.png'}")


if __name__ == "__main__":
    main()
