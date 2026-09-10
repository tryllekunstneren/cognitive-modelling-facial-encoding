"""
Experiment 2: rate the 11 corrected synthetic images, >=10 times each
(exercise step 7).

Validates the encoding model: participants rate the synthetic faces from
data/synthetic_corrected/ on the same 1-5 happy/sad scale as Experiment 1.
Each of the 11 images is shown 10 times (110 trials total), fully
randomised order.

Writes:
  data/experiment2/<student_id>_trials.csv  - one row per trial:
      presentation_order, filename, target_rating, rating, rt_seconds
  (target_rating is the rating the synthetic image was generated for -
  i.e. the model's own prediction for that image, by construction.)

Run:  python scripts/11_experiment2.py
Each group member runs this once, entering their own student ID.
"""

import random
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog

BASE = Path(__file__).resolve().parent.parent
SYN_DIR = BASE / "data" / "synthetic_corrected"
MANIFEST_CSV = SYN_DIR / "synthetic_corrected_manifest.csv"
OUT_DIR = BASE / "data" / "experiment2"
DISPLAY_SIZE = 480
REPEATS_PER_IMAGE = 10

SCALE_TEXT = "1 = very sad        2 = sad        3 = neutral        4 = happy        5 = very happy"


class Experiment2App:
    def __init__(self, root, student_id, trial_specs):
        self.root = root
        self.student_id = student_id
        self.root.title(f"Experiment 2 - {student_id}")

        trials = list(trial_specs) * REPEATS_PER_IMAGE
        random.shuffle(trials)
        self.trials = trials
        self.idx = 0
        self.results = []

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.scale_label = tk.Label(root, text=SCALE_TEXT, font=("Segoe UI", 12))
        self.scale_label.pack(pady=6)

        self.status_label = tk.Label(root, font=("Segoe UI", 10), fg="gray")
        self.status_label.pack()

        for k in "12345":
            root.bind(k, self.make_handler(int(k)))
        root.bind("<Escape>", lambda e: self.quit_early())

        self.trial_start = None
        self.show_current()

    def make_handler(self, rating):
        def handler(event):
            self.record(rating)
        return handler

    def show_current(self):
        if self.idx >= len(self.trials):
            self.finish()
            return
        filename, target_rating = self.trials[self.idx]
        img = Image.open(SYN_DIR / filename)
        img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)
        self.status_label.config(
            text=f"Trial {self.idx + 1} / {len(self.trials)}   "
                 "(press 1-5 to rate, Esc to stop)"
        )
        self.trial_start = time.time()

    def record(self, rating):
        if self.idx >= len(self.trials):
            return
        filename, target_rating = self.trials[self.idx]
        rt = time.time() - self.trial_start
        self.results.append({
            "presentation_order": self.idx + 1,
            "filename": filename,
            "target_rating": target_rating,
            "rating": rating,
            "rt_seconds": round(rt, 3),
        })
        self.idx += 1
        self.show_current()

    def quit_early(self):
        self.save()
        self.root.destroy()

    def finish(self):
        self.img_label.config(image="", text="Thank you! Experiment 2 complete.",
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
    if not MANIFEST_CSV.exists():
        raise SystemExit(
            f"{MANIFEST_CSV} not found. Run scripts/10_range_check.py first."
        )
    manifest = pd.read_csv(MANIFEST_CSV)
    trial_specs = list(zip(manifest["filename"], manifest["target_rating"]))

    root = tk.Tk()
    root.withdraw()
    student_id = simpledialog.askstring(
        "Student ID", "Enter your student ID (used as the output filename):"
    )
    if not student_id:
        raise SystemExit("No student ID entered, aborting.")
    student_id = student_id.strip()
    root.deiconify()

    Experiment2App(root, student_id, trial_specs)
    root.mainloop()


if __name__ == "__main__":
    main()
