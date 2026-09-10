# Encoding of Facial Features (Cognitive Modelling, exercise 2.5.1)

Linear encoding/decoding model of facial expression (happy/sad) built from
UTKFace faces (age 18-25, white, female), following Fagertun et al.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3 with Tk support (bundled by default on Windows/macOS; on
Linux install `python3-tk` if `import tkinter` fails).

## Repo layout

- `scripts/` - pipeline scripts, run in numeric order
- `data/final/` - the 200 processed (grayscale, cropped, 128x128) images used
  in the experiments. Committed to the repo, so you do **not** need to
  download the raw UTKFace dataset just to run Experiment 1.
- `data/final_manifest.csv` - filename/age/gender manifest for the 200 images
- `data/raw/` - the original UTKFace dataset (gitignored - only needed if
  you're redoing image selection from scratch; download from
  https://www.kaggle.com/datasets/abhikjha/utk-face-cropped)
- `data/experiment1/` - per-participant rating data (gitignored is NOT
  applied here, so ratings get committed once produced)

## Running Experiment 1 (rate the 200 images)

Each group member runs this once, on their own machine, entering their own
student ID when prompted:

```bash
python scripts/5_experiment1.py
```

Each image is shown twice (400 trials total, fully randomised order). Rate
each face 1-5 (1 = very sad, 5 = very happy) using the number keys. This
writes `data/experiment1/<student_id>.csv` (the file to submit) and
`data/experiment1/<student_id>_trials.csv` (raw trial log).

Commit and push your two output files afterwards so the group can pool the
data:

```bash
git add data/experiment1/<student_id>.csv data/experiment1/<student_id>_trials.csv
git commit -m "Add <student_id> Experiment 1 ratings"
git push
```

## Pipeline status

1. `2_filter_candidates.py` - filter UTKFace by age/race/gender -> `data/candidates.csv` (done)
2. `3_review_gui.py` - manual keep/discard QC -> `data/review.csv` (done)
3. `4_process_final.py` - trim to 200, grayscale + crop + downsample -> `data/final/` (done)
4. `5_experiment1.py` - the happy/sad rating experiment (run by each group member)
5. `6_analyze_experiment1.py` - per-participant histograms + normalisation check

Steps beyond this (PCA/encoding model, synthetic image generation,
Experiment 2, Experiment 3) are not yet implemented.
