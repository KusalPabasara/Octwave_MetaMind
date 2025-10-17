# Best-So-Far (ResNet50 HIGH_RISK_PLUS) Inference

This folder contains the exact inference setup used to generate our best public leaderboard score:

- Submission file: `submission_resnet50_HIGH_RISK_PLUS.csv`
- Leaderboard score: 0.58661
- Model: 6‑channel ResNet50 (two RGB frames stacked) with three heads (added, removed, changed)
- Threshold preset (HIGH_RISK_PLUS): added=0.48, removed=0.68, changed=0.70

## What’s here

- `run.py` — wrapper that sets environment variables to the HIGH_RISK_PLUS defaults and launches `test.py`.
- `test.py` — inference script that loads the model and label encoders, runs prediction over `test.csv`, and writes the submission CSV.

Required assets at repository root:
- `resnet50_f1_optimized.pth` — trained model weights
- `label_encoders_resnet50.pth` — MultiLabelBinarizers for added/removed/changed
- `test.csv` — test metadata with a column `img_id`
- `data/data/` — images named `{img_id}_1.png` and `{img_id}_2.png`

Output:
- `submission_resnet50_HIGH_RISK_PLUS.csv` — columns: `img_id,added_objs,removed_objs,changed_objs`

## Quick start

1) Install dependencies (Python environment activated):

```bash
pip install torch torchvision pandas numpy pillow tqdm
```

2) From the repository root, run the wrapper (recommended):

```bash
python best_so_far1.0/run.py
```

3) Or run the inference script directly (uses the same default env vars):

```bash
python best_so_far1.0/test.py
```

You should see device info, test sample count, and a message confirming the saved submission at the end.

Verification:
- Line count should be 1,483 (1 header + 1,482 predictions)
- Columns: `img_id, added_objs, removed_objs, changed_objs`

## Configuration (environment variables)

Override any of these to customize paths or thresholds. `run.py` provides defaults tuned for the best score.

- `DATA_DIR` (default: `<repo>/data/data`)
- `TEST_CSV` (default: `<repo>/test.csv`)
- `MODEL_PATH` (default: `<repo>/resnet50_f1_optimized.pth`)
- `LABEL_ENCODERS` (default: `<repo>/label_encoders_resnet50.pth`)
- `SUBMISSION_PATH` (default: `<repo>/submission_resnet50_HIGH_RISK_PLUS.csv`)
- `BATCH_SIZE` (default: `16`)
- `IMAGE_SIZE` (default: `224`)
- `NUM_WORKERS` (default: `4`)
- `THRESH_ADDED` (default: `0.48`)
- `THRESH_REMOVED` (default: `0.68`)
- `THRESH_CHANGED` (default: `0.70`)

Example (custom output + tweaked thresholds):

```bash
SUBMISSION_PATH=./my_submission.csv \
THRESH_ADDED=0.50 THRESH_REMOVED=0.70 THRESH_CHANGED=0.72 \
python best_so_far1.0/test.py
```

## How it works (overview)

- Two input images are resized/normalized and concatenated along the channel dimension → 6‑channel tensor.
- ResNet50 backbone processes features; three linear heads output logits for added/removed/changed.
- Sigmoid → class probabilities → thresholding → decode to class names using saved label encoders.
- Empty sets are written as `none`.

## Tips & troubleshooting

- GPU vs CPU: The script auto‑selects CUDA if available; otherwise runs on CPU. If CUDA isn’t used, check your PyTorch install and drivers.
- Missing files: Ensure `resnet50_f1_optimized.pth`, `label_encoders_resnet50.pth`, `test.csv`, and `data/data/` exist (or point env vars to them).
- Row mismatch: Confirm `test.csv` has 1,482 rows and all images exist for each `img_id`.

## Notes

- The HIGH_RISK_PLUS thresholds produced the best observed public LB score (0.58661) among our ResNet50 runs.
- You can sweep thresholds via env vars without retraining to explore precision/recall trade‑offs.
- This path performs pure per‑head thresholding; it doesn’t force coexistence rules between added/removed/changed.
