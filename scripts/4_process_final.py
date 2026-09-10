"""
Build the final 200-image set from data/review.csv (output of the manual
QC GUI) and process it: grayscale + center-crop + downsample.

- Drops everything tagged "discard".
- If more than 200 remain, randomly trims down to exactly 200.
- Saves processed grayscale images to data/final/, and a manifest
  data/final_manifest.csv (filename, age, gender) as the appendix data
  file referenced in the report. Expression (happy/sad) is NOT recorded
  here - it is only measured later, from the participants' ratings in
  Experiment 1.

Run:  python scripts/4_process_final.py
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent.parent
REVIEW_CSV = BASE / "data" / "review.csv"
FINAL_DIR = BASE / "data" / "final"
MANIFEST_CSV = BASE / "data" / "final_manifest.csv"

N_FINAL = 200
OUT_SIZE = 128  # final square image size in pixels after crop+downsample
RANDOM_SEED = 42


def trim(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    if len(df) <= n_target:
        return df
    return df.sample(n=n_target, random_state=seed)


def process_image(path: Path, out_path: Path, size: int):
    img = Image.open(path).convert("L")  # grayscale
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    img.save(out_path)


def main():
    if not REVIEW_CSV.exists():
        raise SystemExit(
            f"{REVIEW_CSV} not found. Run scripts/3_review_gui.py first "
            "and tag all candidate images."
        )

    df = pd.read_csv(REVIEW_CSV)
    kept = df[df.label == "keep"].copy()
    print(f"Kept after QC: {len(kept)} (discarded: {len(df) - len(kept)})")
    if len(kept) < N_FINAL:
        print(f"WARNING: only {len(kept)} images kept, fewer than the "
              f"target of {N_FINAL}. Consider drawing more candidates "
              "from scripts/2_filter_candidates.py.")

    final = trim(kept, N_FINAL, RANDOM_SEED)
    print(f"\nFinal set size: {len(final)}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for _, row in final.iterrows():
        out_name = row.filename.replace(".chip.jpg", "").replace(".jpg", "") + "_proc.png"
        out_path = FINAL_DIR / out_name
        process_image(Path(row.filepath), out_path, OUT_SIZE)
        manifest_rows.append({
            "processed_filename": out_name,
            "original_filename": row.filename,
            "age": row.age,
            "gender": "F" if row.gender == 1 else "M",
        })

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)
    print(f"\nSaved {len(manifest_rows)} processed images to {FINAL_DIR}")
    print(f"Manifest written to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
