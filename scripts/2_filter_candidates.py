"""
Parse UTKFace filenames and build a balanced candidate list for Experiment 1.

Filename format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
  gender: 0 = male, 1 = female
  race:   0 = white, 1 = black, 2 = asian, 3 = indian, 4 = other

Criteria (fixed for this project):
  - age in [18, 25]
  - race == white
  - gender == female (dataset is female-only; happy/sad rating is the
    only feature that should vary)

Output: data/candidates.csv (full filtered pool) and prints how many
images are available, so we know how large a random ~300 draw can be.
"""

import re
import random
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "utkcropped"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "candidates.csv"

AGE_MIN, AGE_MAX = 18, 25
RACE_WHITE = 0
GENDER_FEMALE = 1
N_TARGET = 300  # target size of the Experiment-1 subset
RANDOM_SEED = 42

FNAME_RE = re.compile(r"^(\d+)_(\d)_(\d)_.*\.jpg(?:\.chip\.jpg)?$", re.IGNORECASE)


def parse_filename(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    age, gender, race = m.groups()
    return int(age), int(gender), int(race)


def main():
    rows = []
    skipped = 0
    for path in RAW_DIR.glob("*.jpg"):
        parsed = parse_filename(path.name)
        if parsed is None:
            skipped += 1
            continue
        age, gender, race = parsed
        rows.append({"filepath": str(path), "filename": path.name,
                      "age": age, "gender": gender, "race": race})

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} files, skipped {skipped} unparseable filenames.")

    pool = df[(df.age >= AGE_MIN) & (df.age <= AGE_MAX) &
              (df.race == RACE_WHITE) & (df.gender == GENDER_FEMALE)]
    print(f"Pool after age {AGE_MIN}-{AGE_MAX} + white + female filter: {len(pool)} images")
    print(pool.groupby("age").size())

    n_sample = min(len(pool), N_TARGET)
    sample = pool.sample(n=n_sample, random_state=RANDOM_SEED)

    sample.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(sample)} candidate images to {OUT_CSV}")
    print("This is your ~300-image pool for Experiment 1, before manual "
          "happy/sad selection and QC.")


if __name__ == "__main__":
    main()
