"""
Experiment 1: rate each final image on a 1-5 happy/sad scale.

Each of the 200 images in data/final/ is shown TWICE, and the order of
all 400 presentations is fully randomised (not grouped by image). The
participant rates each presentation on a 1-5 scale using the number keys.

At the end, one row per image is written with both ratings:
  filename, rating_1, rating_2
to data/experiment1/<student_id>.csv

A raw trial-by-trial log (with presentation order and timestamps) is also
kept in data/experiment1/<student_id>_trials.csv for transparency /
debugging, but the file to submit is the per-image one.

Run:  python scripts/5_experiment1.py
Each group member runs this once, entering their own student ID.
"""

import csv
import random
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog

BASE = Path(__file__).resolve().parent.parent
MANIFEST_CSV = BASE / "data" / "final_manifest.csv"
FINAL_DIR = BASE / "data" / "final"
OUT_DIR = BASE / "data" / "experiment1"
DISPLAY_SIZE = 480

SCALE_TEXT = "1 = very sad        2 = sad        3 = neutral        4 = happy        5 = very happy"


class ExperimentApp:
    def __init__(self, root, student_id, images):
        self.root = root
        self.student_id = student_id
        self.root.title(f"Experiment 1 - {student_id}")

        # build trial list: each image appears twice, order fully randomised
        trials = list(images) * 2
        random.shuffle(trials)
        self.trials = trials
        self.idx = 0
        self.results = []  # list of dicts: presentation_order, filename, rating, rt

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
        filename = self.trials[self.idx]
        img = Image.open(FINAL_DIR / filename)
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
        filename = self.trials[self.idx]
        rt = time.time() - self.trial_start
        self.results.append({
            "presentation_order": self.idx + 1,
            "filename": filename,
            "rating": rating,
            "rt_seconds": round(rt, 3),
        })
        self.idx += 1
        self.show_current()

    def quit_early(self):
        self.save()
        self.root.destroy()

    def finish(self):
        self.img_label.config(image="", text="Thank you! Experiment complete.",
                               font=("Segoe UI", 18))
        self.scale_label.config(text="")
        self.status_label.config(text="")
        self.save()

    def save(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # raw trial log
        trials_path = OUT_DIR / f"{self.student_id}_trials.csv"
        pd.DataFrame(self.results).to_csv(trials_path, index=False)

        # per-image file: filename, rating_1, rating_2 (in presentation order)
        per_image = {}
        for r in self.results:
            per_image.setdefault(r["filename"], []).append(r["rating"])

        out_path = OUT_DIR / f"{self.student_id}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for filename, ratings in per_image.items():
                ratings = ratings + [""] * (2 - len(ratings))  # pad if quit early
                writer.writerow([filename, ratings[0], ratings[1]])

        print(f"Saved {len(per_image)} images' ratings to {out_path}")
        print(f"Raw trial log saved to {trials_path}")


def main():
    if not MANIFEST_CSV.exists():
        raise SystemExit(
            f"{MANIFEST_CSV} not found. Run scripts/4_process_final.py first."
        )
    manifest = pd.read_csv(MANIFEST_CSV)
    images = manifest["processed_filename"].tolist()

    root = tk.Tk()
    root.withdraw()
    student_id = simpledialog.askstring(
        "Student ID", "Enter your student ID (used as the output filename):"
    )
    if not student_id:
        raise SystemExit("No student ID entered, aborting.")
    student_id = student_id.strip()
    root.deiconify()

    ExperimentApp(root, student_id, images)
    root.mainloop()


if __name__ == "__main__":
    main()
