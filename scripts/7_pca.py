"""
PCA and dimension reduction on the 200 final images (exercise step 3).

- Averages each image's Experiment 1 ratings (2 trials x N participants)
  into a single mean happy/sad rating -> data/ratings_mean.csv. This is the
  dependent variable used by the encoding model in later scripts.
- Loads the 200 grayscale 128x128 images, flattens them, subtracts the
  average image (no std-normalisation), and runs PCA on the centered data.
- Saves the mean image, PCA components/scores/explained variance to
  data/pca/ so later scripts (regression, synthetic image generation) don't
  need to redo this.
- Visualises the first 5 PCs as (mean - k*PC, mean, mean + k*PC) triplets
  using each PC's actual min/max score in the dataset.
- Plots variance explained per PC (scree plot).

Run:  python scripts/7_pca.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE / "data" / "final"
MANIFEST_CSV = BASE / "data" / "final_manifest.csv"
EXP_DIR = BASE / "data" / "experiment1"
PCA_DIR = BASE / "data" / "pca"
PLOTS_DIR = PCA_DIR / "plots"

N_PCS_TO_VISUALISE = 5
VARIANCE_TARGET = 0.95  # keep enough PCs to explain this much variance


def load_ratings() -> pd.DataFrame:
    """Average all ratings per image, across both trials and all
    participants, into one mean rating. Uses the *_normalised.csv file for
    a participant if one was written (i.e. that participant didn't use the
    full 1-5 range), otherwise their raw file."""
    participant_files = {}
    for path in EXP_DIR.glob("*.csv"):
        if path.name.endswith("_trials.csv"):
            continue
        student_id = path.stem.replace("_normalised", "")
        # prefer the normalised version if it exists
        participant_files[student_id] = path
    for student_id in list(participant_files):
        norm_path = EXP_DIR / f"{student_id}_normalised.csv"
        if norm_path.exists():
            participant_files[student_id] = norm_path

    all_ratings = []
    for student_id, path in participant_files.items():
        df = pd.read_csv(path, header=None, names=["filename", "rating_1", "rating_2"])
        df["rating_1"] = pd.to_numeric(df.rating_1, errors="coerce")
        df["rating_2"] = pd.to_numeric(df.rating_2, errors="coerce")
        all_ratings.append(df)

    combined = pd.concat(all_ratings, ignore_index=True)
    mean_ratings = (
        combined.melt(id_vars="filename", value_vars=["rating_1", "rating_2"], value_name="rating")
        .groupby("filename")["rating"].mean()
        .reset_index()
        .rename(columns={"rating": "mean_rating"})
    )
    print(f"Averaged ratings from {len(participant_files)} participant(s): "
          f"{', '.join(participant_files)}")
    return mean_ratings


def load_images(filenames):
    vectors = []
    shape = None
    for fname in filenames:
        img = Image.open(FINAL_DIR / fname)
        arr = np.asarray(img, dtype=np.float64)
        if shape is None:
            shape = arr.shape
        vectors.append(arr.flatten())
    return np.vstack(vectors), shape


def make_pc_triplet(mean_image, pc_vector, min_score, max_score, shape):
    lo = (mean_image + min_score * pc_vector).reshape(shape)
    mid = mean_image.reshape(shape)
    hi = (mean_image + max_score * pc_vector).reshape(shape)
    return lo, mid, hi


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    manifest = pd.read_csv(MANIFEST_CSV)
    ratings = load_ratings()

    merged = manifest.merge(ratings, left_on="processed_filename", right_on="filename", how="inner")
    missing = set(manifest.processed_filename) - set(merged.processed_filename)
    if missing:
        print(f"WARNING: {len(missing)} images have no rating data and were dropped: "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    merged = merged.sort_values("processed_filename").reset_index(drop=True)
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    merged[["processed_filename", "mean_rating"]].to_csv(
        BASE / "data" / "ratings_mean.csv", index=False)
    print(f"Wrote mean ratings for {len(merged)} images to data/ratings_mean.csv")

    X, img_shape = load_images(merged["processed_filename"])
    print(f"Loaded {X.shape[0]} images, {X.shape[1]} pixels each (shape {img_shape})")

    mean_image = X.mean(axis=0)
    X_centered = X - mean_image

    n_components = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_centered)  # (n_images, n_components)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum_var, VARIANCE_TARGET) + 1)
    n_keep = min(n_keep, n_components)
    print(f"\n{n_components} PCs total. Top {n_keep} PCs explain "
          f"{cum_var[n_keep - 1]*100:.1f}% of variance "
          f"(target: {VARIANCE_TARGET*100:.0f}%).")

    # Save everything downstream scripts need
    np.save(PCA_DIR / "mean_image.npy", mean_image)
    np.save(PCA_DIR / "components.npy", pca.components_)  # (n_components, n_pixels)
    np.save(PCA_DIR / "explained_variance_ratio.npy", pca.explained_variance_ratio_)
    with open(PCA_DIR / "image_shape.txt", "w") as f:
        f.write(f"{img_shape[0]},{img_shape[1]}")

    scores_df = pd.DataFrame(
        scores, columns=[f"PC{i+1}" for i in range(n_components)])
    scores_df.insert(0, "processed_filename", merged["processed_filename"].values)
    scores_df.insert(1, "mean_rating", merged["mean_rating"].values)
    scores_df.to_csv(PCA_DIR / "scores.csv", index=False)

    with open(PCA_DIR / "n_pcs_recommended.txt", "w") as f:
        f.write(str(n_keep))

    print(f"Saved mean image, components, scores, and recommended n_keep={n_keep} to {PCA_DIR}")

    # --- visualisations ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(N_PCS_TO_VISUALISE, 3, figsize=(6, 2 * N_PCS_TO_VISUALISE))
    for i in range(N_PCS_TO_VISUALISE):
        pc_vector = pca.components_[i]
        pc_scores = scores[:, i]
        lo, mid, hi = make_pc_triplet(
            mean_image, pc_vector, pc_scores.min(), pc_scores.max(), img_shape)
        for ax, img, title in zip(axes[i], [lo, mid, hi],
                                   [f"PC{i+1} min", "average", f"PC{i+1} max"]):
            ax.imshow(to_uint8(img), cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pc_triplets.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    n_show = min(30, n_components)
    ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100,
           color="#4C72B0")
    ax.axvline(n_keep + 0.5, color="crimson", linestyle="--",
               label=f"kept: {n_keep} PCs ({VARIANCE_TARGET*100:.0f}% variance)")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Scree plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scree_plot.png", dpi=150)
    plt.close(fig)

    print(f"Saved PC triplet figure and scree plot to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
