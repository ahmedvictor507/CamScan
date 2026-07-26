# CamScan

A CamScanner-style document scanner: takes a photo of a document against an arbitrary
background and outputs a clean, flattened, enhanced scan.

The core challenge is **document boundary detection**. This project builds and compares
multiple classical CV approaches (plus a learned-segmentation stretch goal) rather than
shipping a single naive method.

## Pipeline

1. Preprocess (grayscale, blur)
2. Edge detection (Canny)
3. Document boundary detection — compared methods:
   - Baseline: largest contour + 4-point polygon approximation
   - + aspect-ratio-constrained candidates (A4/Letter)
   - + inside/outside contrast scoring of candidate quadrilaterals
   - Stretch: SAM box-prompt segmentation as a learned-method comparison point
4. Perspective transform (4-point warp to a flat rectangle)
5. Enhancement (adaptive thresholding / contrast / deskew)
6. Output the processed scan

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Test set

`data/raw/` holds ~15-20 self-collected photos across four conditions: `clean/`,
`cluttered/`, `low_light/`, `skewed/`. `data/results/` holds per-method outputs used
for comparison.

## Progress log

See `docs/progress_log.md`.
