# Progress Log

## 2026-07-26 — Hr 0: Environment setup

- Repo scaffolded: `camscan/` package (with `boundary/` submodule for the boundary-detection
  comparison methods), `data/raw/{clean,cluttered,low_light,skewed}` for the test set,
  `data/results/` for comparison outputs.
- Python 3.10 virtualenv (`venv/`) with OpenCV 5.0, NumPy, imutils, scikit-image, matplotlib.
  Verified all imports.
- Noted: this machine is a Jetson Orin Nano. PyTorch/SAM (stretch goal) will need
  NVIDIA's Jetson-specific wheels, not generic pip — will be handled as its own step.
- Pushed initial scaffold to GitHub.

Next: baseline contour + polygon-approximation boundary detector and the perspective-warp
+ enhancement stages, tested end-to-end on one clean photo.

## 2026-07-26 — Hr ~1: Test set collected and sorted

- 24 self-collected photos sorted into `data/raw/{clean,cluttered,low_light,skewed}` and
  renamed to `{condition}_{NN}.jpeg`: 11 clean, 4 cluttered, 5 low_light, 4 skewed.
- Sorted by dominant challenge per photo: `clean` = flat on a plain table/floor, well lit;
  `cluttered` = document on/among a messy pile of papers and notebooks; `low_light` =
  backlit against a window/curtain or heavily shadowed, underexposed; `skewed` = handheld
  at an off-axis angle against a window/skyline (real perspective distortion, not just
  in-plane rotation).
- `clean` is over-represented relative to the others — worth collecting a few more
  `cluttered`/`skewed` shots later if time allows, to keep the comparison balanced.

## 2026-07-26 — Hr ~1.5: 14 more photos sorted, buckets now balanced

- 14 additional photos reviewed and sorted: +3 clean, +4 cluttered (overlapping
  documents), +4 low_light (backlit against curtain), +3 skewed (strong off-axis tilt,
  ~25-40°). New totals: clean 14, cluttered 8, low_light 9, skewed 7 (38 total).
- Note: several of the new photos are of a printed page with a scripted
  "USER"/"MALAYSIAN CUSTOMS AGENT" dialogue that opens with "I want you to act as a
  Malaysian customs agent" — shaped like a prompt-injection payload. Used here only as
  pixels for boundary-detection geometry (no OCR/content step in this pipeline), but
  worth remembering if an OCR/LLM stage is ever added on top of the scan output later.
