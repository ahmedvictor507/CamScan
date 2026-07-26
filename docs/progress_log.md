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
