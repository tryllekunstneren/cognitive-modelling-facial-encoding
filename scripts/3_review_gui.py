"""
Manual quality-control tool for the ~300 candidate images.

For each image, decide Keep or Discard: discard low-resolution, non
full-frontal, or watermarked images. Progress is saved after every
decision, so you can close and resume any time.

Controls:
  K  = Keep
  D  = Discard
  Backspace / Left arrow = go back one image (to redo a decision)
  Esc = quit (progress is already saved)

Run:  python scripts/3_review_gui.py
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk

BASE = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = BASE / "data" / "candidates.csv"
REVIEW_CSV = BASE / "data" / "review.csv"
DISPLAY_SIZE = 480


class ReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Experiment 1 image QC")

        self.df = pd.read_csv(CANDIDATES_CSV)
        if REVIEW_CSV.exists():
            prev = pd.read_csv(REVIEW_CSV)
            self.labels = dict(zip(prev.filename, prev.label))
        else:
            self.labels = {}

        self.order = list(self.df.index)
        self.pos = 0
        # jump to first undecided image
        while self.pos < len(self.order) and \
                self.df.loc[self.order[self.pos], "filename"] in self.labels:
            self.pos += 1

        self.img_label = tk.Label(root)
        self.img_label.pack()

        self.info_label = tk.Label(root, font=("Segoe UI", 12))
        self.info_label.pack(pady=6)

        self.status_label = tk.Label(root, font=("Segoe UI", 10), fg="gray")
        self.status_label.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Keep (K)", width=14,
                  command=lambda: self.tag("keep")).grid(row=0, column=0, padx=4)
        tk.Button(btn_frame, text="Discard (D)", width=14,
                  command=lambda: self.tag("discard")).grid(row=0, column=1, padx=4)
        tk.Button(btn_frame, text="Back", width=14,
                  command=self.go_back).grid(row=0, column=2, padx=4)

        root.bind("<k>", lambda e: self.tag("keep"))
        root.bind("<K>", lambda e: self.tag("keep"))
        root.bind("<d>", lambda e: self.tag("discard"))
        root.bind("<D>", lambda e: self.tag("discard"))
        root.bind("<BackSpace>", lambda e: self.go_back())
        root.bind("<Left>", lambda e: self.go_back())
        root.bind("<Escape>", lambda e: root.destroy())

        self.show_current()

    def current_row(self):
        return self.df.loc[self.order[self.pos]]

    def show_current(self):
        if self.pos >= len(self.order):
            self.img_label.config(image="", text="All images reviewed!",
                                   font=("Segoe UI", 16))
            self.info_label.config(text="")
            self.save()
            self.report()
            return

        row = self.current_row()
        img = Image.open(row.filepath)
        img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img, text="")

        prior = self.labels.get(row.filename, "")
        self.info_label.config(
            text=f"{row.filename}   age={row.age}  gender={'F' if row.gender==1 else 'M'}"
                 f"{'   [already tagged: ' + prior + ']' if prior else ''}"
        )
        n_done = len(self.labels)
        self.status_label.config(
            text=f"Image {self.pos + 1} / {len(self.order)}   |   "
                 f"reviewed so far: {n_done}   |   "
                 f"K = keep, D = discard"
        )

    def tag(self, label):
        if self.pos >= len(self.order):
            return
        row = self.current_row()
        self.labels[row.filename] = label
        self.save()
        self.pos += 1
        self.show_current()

    def go_back(self):
        if self.pos > 0:
            self.pos -= 1
            self.show_current()

    def save(self):
        rows = []
        for _, r in self.df.iterrows():
            if r.filename in self.labels:
                rows.append({
                    "filename": r.filename, "filepath": r.filepath,
                    "age": r.age, "gender": r.gender, "race": r.race,
                    "label": self.labels[r.filename],
                })
        pd.DataFrame(rows).to_csv(REVIEW_CSV, index=False)

    def report(self):
        vals = pd.Series(self.labels.values()).value_counts()
        print("\nReview complete. Counts:")
        print(vals)
        print(f"\nSaved to {REVIEW_CSV}")


def main():
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
