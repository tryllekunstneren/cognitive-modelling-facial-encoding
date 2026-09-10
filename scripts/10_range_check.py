"""
Range check for the synthetic images (exercise step 6).

The model predicts a rating for every one of the 200 training images via
rating = intercept + w . scores. The min and max of those predictions
define the range of ratings the model actually supports - outside that
range, "generating" a rating means extrapolating along w with a huge
alpha (see scripts/9_synthetic_images.py), which can push pixel values
far outside [0, 255] and produce unnatural artifacts.

This script:
  1. Recomputes that predicted-rating range on the 200 training images.
  2. Compares it to the 0.5-5.5 range used in scripts/9_synthetic_images.py.
  3. Since they deviate substantially, generates a NEW set of 11 synthetic
     images with target ratings evenly spaced *within* the model's
     supported range instead, and saves them to data/synthetic_corrected/.

Run:  python scripts/10_range_check.py
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
OLD_SYN_DIR = BASE / "data" / "synthetic"
SYN_DIR = BASE / "data" / "synthetic_corrected"
PLOTS_DIR = SYN_DIR / "plots"

N_SYNTHETIC = 11
ORIGINAL_GEN_MIN, ORIGINAL_GEN_MAX = 0.5, 5.5


def main():
    mean_image = np.load(PCA_DIR / "mean_image.npy")
    components = np.load(PCA_DIR / "components.npy")
    shape = tuple(int(x) for x in (PCA_DIR / "image_shape.txt").read_text().split(","))

    selected_idx = np.load(REG_DIR / "selected_pc_indices.npy")
    coef = np.load(REG_DIR / "model_coef.npy")
    intercept = float((REG_DIR / "model_intercept.txt").read_text())
    w_pixel = coef @ components[selected_idx]
    w_normsq = float(np.dot(coef, coef))

    scores_df = pd.read_csv(PCA_DIR / "scores.csv")
    pc_cols = [f"PC{i+1}" for i in selected_idx]
    X_sel = scores_df[pc_cols].values
    predicted = intercept + X_sel @ coef
    pred_min, pred_max = predicted.min(), predicted.max()

    print(f"Predicted rating range on the 200 training images: [{pred_min:.3f}, {pred_max:.3f}]")
    print(f"Range used to generate synthetic images in step 5: "
          f"[{ORIGINAL_GEN_MIN}, {ORIGINAL_GEN_MAX}]")

    deviates = (ORIGINAL_GEN_MIN < pred_min - 0.2) or (ORIGINAL_GEN_MAX > pred_max + 0.2)
    if not deviates:
        print("Ranges are close enough - no need to regenerate synthetic images.")
        return

    print("\nThe step-5 generation range extends well beyond what the model ever "
          "predicts for a real training image - i.e. it extrapolates. "
          "Regenerating synthetic images confined to the model's predicted range.")

    targets = np.linspace(pred_min, pred_max, N_SYNTHETIC)

    SYN_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    images_by_rating = {}
    for target in targets:
        alpha = (target - intercept) / w_normsq
        img = (mean_image + alpha * w_pixel).reshape(shape)
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        images_by_rating[target] = img_clipped

        out_name = f"synthetic_corrected_rating_{target:.2f}.png"
        Image.fromarray(img_clipped).save(SYN_DIR / out_name)
        manifest_rows.append({
            "target_rating": target,
            "alpha": alpha,
            "filename": out_name,
            "pixel_min_before_clip": float(img.min()),
            "pixel_max_before_clip": float(img.max()),
        })
        print(f"target={target:.2f}  alpha={alpha:+.2f}  "
              f"pixel range before clipping=[{img.min():.0f}, {img.max():.0f}]")

    pd.DataFrame(manifest_rows).to_csv(SYN_DIR / "synthetic_corrected_manifest.csv", index=False)

    sorted_ratings = sorted(images_by_rating)
    fig, axes = plt.subplots(1, len(sorted_ratings), figsize=(2 * len(sorted_ratings), 2.4))
    for ax, target in zip(axes, sorted_ratings):
        ax.imshow(images_by_rating[target], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{target:.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Corrected synthetic faces (within model's predicted rating range)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "synthetic_faces_corrected_row.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved corrected synthetic images + manifest to {SYN_DIR}")
    print(f"Saved composite figure to {PLOTS_DIR / 'synthetic_faces_corrected_row.png'}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Report answer - "Describe the min/max predicted rating for the training
# stimuli. Evaluate whether they deviate substantially from the min/max
# ratings used to create your stimuli. If so, create and present a new set
# of synthetic images."
#
# The model's predicted ratings on the 200 training images range from 2.30
# to 5.39. The step-5 synthetic images were generated across 0.5-5.5 - a
# substantial deviation, especially at the low end (0.5 vs. 2.30), because
# no real training image ever produced a predicted rating anywhere near
# "very sad". This matches the clipping artifact seen in step 5's lowest
# two faces: those targets required extrapolating far outside the data.
#
# A corrected set of 11 synthetic images was generated instead, evenly
# spaced within [2.30, 5.39] (data/synthetic_corrected/plots/
# synthetic_faces_corrected_row.png). None of these show pixel values
# outside [0, 255] before clipping (see synthetic_corrected_manifest.csv),
# and the mouth still transitions smoothly from flatter/neutral to a clear
# smile - this corrected set is the one that should be used/presented
# going forward (e.g. for Experiment 2 stimuli).
# ---------------------------------------------------------------------------
