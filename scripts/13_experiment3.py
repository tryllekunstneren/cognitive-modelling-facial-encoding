"""
Experiment 3: perceptual after-effect (exercise step 8).

Tests whether adapting to a synthetic face at one endpoint of the
happy/sad continuum shifts perception of near-neutral test faces away
from that endpoint (a contrastive after-effect).

Stimuli (5 unique images, all with a central fixation point):
  - 2 adapting stimuli: the two endpoints of data/synthetic_corrected/
    (ratings 2.30 "sad end" and 5.39 "happy end")
  - 3 test stimuli: newly generated, close to the model's neutral rating
    (targets 2.8, 3.0, 3.2 - well inside the model's supported
    [2.30, 5.39] range, so no extrapolation risk, see step 6)

Design: 2 adaptation conditions x 3 test stimuli = 6 conditions,
repeated REPEATS_PER_CONDITION times each (shuffled trial order).

Each trial:
  1. Adapting stimulus + fixation point, shown for a random 20-30s
  2. Immediately, test stimulus + fixation point, shown for a random
     0.5-1.0s
  3. Rating prompt (1-5, same scale as Experiments 1/2), untimed

Writes data/experiment3/<student_id>_trials.csv:
  presentation_order, adapt_label, adapt_rating, test_target_rating,
  adapt_duration_s, test_duration_s, response, rt_seconds

Run:  python scripts/13_experiment3.py
Each group member runs this once, entering their own student ID.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import simpledialog

BASE = Path(__file__).resolve().parent.parent
PCA_DIR = BASE / "data" / "pca"
REG_DIR = BASE / "data" / "regression"
CORRECTED_DIR = BASE / "data" / "synthetic_corrected"
STIM_DIR = BASE / "data" / "experiment3" / "stimuli"
OUT_DIR = BASE / "data" / "experiment3"

DISPLAY_SIZE = 480
FIXATION_RADIUS = 4
FIXATION_ARM = 12
FIXATION_COLOR = (255, 40, 40)

TEST_TARGET_RATINGS = [2.8, 3.0, 3.2]
ADAPT_DURATION_RANGE = (20.0, 30.0)   # seconds
TEST_DURATION_RANGE = (0.5, 1.0)      # seconds
REPEATS_PER_CONDITION = 3             # -> 6 conditions x 3 = 18 trials

SCALE_TEXT = "1 = very sad        2 = sad        3 = neutral        4 = happy        5 = very happy"


def generate_neutral_test_stimuli():
    """Create the 3 near-neutral test images if they don't already exist."""
    STIM_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = STIM_DIR / "test_stimuli_manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    mean_image = np.load(PCA_DIR / "mean_image.npy")
    components = np.load(PCA_DIR / "components.npy")
    shape = tuple(int(x) for x in (PCA_DIR / "image_shape.txt").read_text().split(","))
    selected_idx = np.load(REG_DIR / "selected_pc_indices.npy")
    coef = np.load(REG_DIR / "model_coef.npy")
    intercept = float((REG_DIR / "model_intercept.txt").read_text())
    w_pixel = coef @ components[selected_idx]
    w_normsq = float(np.dot(coef, coef))

    rows = []
    for target in TEST_TARGET_RATINGS:
        alpha = (target - intercept) / w_normsq
        img = (mean_image + alpha * w_pixel).reshape(shape)
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        out_name = f"test_{target:.1f}.png"
        Image.fromarray(img_clipped).save(STIM_DIR / out_name)
        rows.append({"test_target_rating": target, "filename": out_name})
        print(f"Generated test stimulus target={target:.1f} -> {out_name}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def with_fixation(path: Path) -> ImageTk.PhotoImage:
    img = Image.open(path).convert("RGB").resize(
        (DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    cx, cy = DISPLAY_SIZE // 2, DISPLAY_SIZE // 2
    draw.line((cx - FIXATION_ARM, cy, cx + FIXATION_ARM, cy), fill=FIXATION_COLOR, width=3)
    draw.line((cx, cy - FIXATION_ARM, cx, cy + FIXATION_ARM), fill=FIXATION_COLOR, width=3)
    return ImageTk.PhotoImage(img)


def build_trial_list(adapt_specs, test_specs):
    conditions = [(a, t) for a in adapt_specs for t in test_specs]
    trials = conditions * REPEATS_PER_CONDITION
    random.shuffle(trials)
    return trials


class Experiment3App:
    def __init__(self, root, student_id, trials, images):
        self.root = root
        self.student_id = student_id
        self.root.title(f"Experiment 3 - {student_id}")
        self.trials = trials
        self.images = images  # dict: filename -> PhotoImage (with fixation)
        self.idx = 0
        self.results = []

        self.img_label = tk.Label(root, bg="gray20")
        self.img_label.pack(pady=10)

        self.scale_label = tk.Label(root, text="", font=("Segoe UI", 12))
        self.scale_label.pack(pady=6)

        self.status_label = tk.Label(root, font=("Segoe UI", 10), fg="gray")
        self.status_label.pack()

        self.awaiting_response = False
        for k in "12345":
            root.bind(k, self.make_handler(int(k)))
        root.bind("<Escape>", lambda e: self.quit_early())

        self.next_trial()

    def make_handler(self, rating):
        def handler(event):
            if self.awaiting_response:
                self.record(rating)
        return handler

    def next_trial(self):
        if self.idx >= len(self.trials):
            self.finish()
            return
        adapt, test = self.trials[self.idx]
        self.current_adapt = adapt
        self.current_test = test
        self.awaiting_response = False
        self.scale_label.config(text="")

        adapt_duration = random.uniform(*ADAPT_DURATION_RANGE)
        self.test_duration = random.uniform(*TEST_DURATION_RANGE)
        self.trial_t0 = time.time()

        self.img_label.config(image=self.images[adapt["filename"]])
        self.status_label.config(
            text=f"Trial {self.idx + 1} / {len(self.trials)}   "
                 f"adapting ({adapt_duration:.0f}s)..."
        )
        self.adapt_duration_used = adapt_duration
        self.root.after(int(adapt_duration * 1000), self.show_test)

    def show_test(self):
        self.img_label.config(image=self.images[self.current_test["filename"]])
        self.status_label.config(text=f"Trial {self.idx + 1} / {len(self.trials)}   test stimulus")
        self.root.after(int(self.test_duration * 1000), self.show_prompt)

    def show_prompt(self):
        self.img_label.config(image="", bg="gray20")
        self.scale_label.config(text=SCALE_TEXT)
        self.status_label.config(
            text=f"Trial {self.idx + 1} / {len(self.trials)}   press 1-5 to rate the last face"
        )
        self.awaiting_response = True
        self.response_t0 = time.time()

    def record(self, rating):
        rt = time.time() - self.response_t0
        self.results.append({
            "presentation_order": self.idx + 1,
            "adapt_label": self.current_adapt["label"],
            "adapt_rating": self.current_adapt["rating"],
            "test_target_rating": self.current_test["test_target_rating"],
            "adapt_duration_s": round(self.adapt_duration_used, 2),
            "test_duration_s": round(self.test_duration, 2),
            "response": rating,
            "rt_seconds": round(rt, 3),
        })
        self.idx += 1
        self.next_trial()

    def quit_early(self):
        self.save()
        self.root.destroy()

    def finish(self):
        self.img_label.config(image="", text="Thank you! Experiment 3 complete.",
                               font=("Segoe UI", 18))
        self.scale_label.config(text="")
        self.status_label.config(text="")
        self.save()

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{self.student_id}_trials.csv"
        pd.DataFrame(self.results).to_csv(out_path, index=False)
        print(f"Saved {len(self.results)} trials to {out_path}")


def main():
    if not (CORRECTED_DIR / "synthetic_corrected_manifest.csv").exists():
        raise SystemExit("Run scripts/10_range_check.py first (need the corrected "
                          "synthetic endpoint images).")

    corrected = pd.read_csv(CORRECTED_DIR / "synthetic_corrected_manifest.csv")
    corrected = corrected.sort_values("target_rating")
    low_row = corrected.iloc[0]
    high_row = corrected.iloc[-1]
    adapt_specs = [
        {"label": "sad_endpoint", "rating": low_row.target_rating,
         "filename": str((CORRECTED_DIR / low_row.filename).resolve())},
        {"label": "happy_endpoint", "rating": high_row.target_rating,
         "filename": str((CORRECTED_DIR / high_row.filename).resolve())},
    ]

    test_manifest = generate_neutral_test_stimuli()
    test_specs = [
        {"test_target_rating": row.test_target_rating,
         "filename": str((STIM_DIR / row.filename).resolve())}
        for _, row in test_manifest.iterrows()
    ]

    trials = build_trial_list(adapt_specs, test_specs)
    print(f"{len(trials)} trials ({len(adapt_specs)} adaptation endpoints x "
          f"{len(test_specs)} test stimuli x {REPEATS_PER_CONDITION} repeats)")

    root = tk.Tk()
    root.withdraw()
    student_id = simpledialog.askstring(
        "Student ID", "Enter your student ID (used as the output filename):"
    )
    if not student_id:
        raise SystemExit("No student ID entered, aborting.")
    student_id = student_id.strip()
    root.deiconify()

    # pre-render all 5 unique images (with fixation point) once - needs a
    # Tk root to already exist, so this happens after root = tk.Tk()
    images = {}
    for spec in adapt_specs + test_specs:
        images[spec["filename"]] = with_fixation(Path(spec["filename"]))

    Experiment3App(root, student_id, trials, images)
    root.mainloop()


if __name__ == "__main__":
    main()
