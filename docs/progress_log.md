# Progress Log

Logged continuously as work happened, timestamped by elapsed hour rather than a fixed
interval — here's how those entries map onto rough 3-hour checkpoints, for a fast skim:

- **Hours 0–3:** environment setup, 38-photo test set collected and sorted across 4
  conditions (clean/cluttered/low_light/skewed), baseline detection pipeline working
  end-to-end.
- **Hours 3–6:** two classical-method improvements (aspect-ratio constraint, boundary
  contrast scoring) and the batch comparison harness (`compare.py`) built to score
  every method against the same 38 images.
- **Hours 6–9:** SAM (MobileSAM) added as a learned-segmentation comparison point,
  including a second-prompt variant that was tried and reverted after a full audit
  showed it made things worse, not better.
- **Hours 9+ / day 2:** five YOLO-pose training attempts (one a documented, honest
  failure), fallback architecture reworked from "first success wins" to
  score-and-pick, production model exported to ONNX and rewritten to drop the
  `ultralytics`/torch dependency (5.92GB → 679MB deploy image), deployed to Railway +
  Vercel, pytest suite added, full type-hint pass.

Full detail, reasoning, and dead ends below.

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

## 2026-07-26 — Hr ~2.5: Baseline pipeline working end-to-end

- Implemented `camscan/{preprocess,edges,warp,enhance,pipeline}.py` and
  `camscan/boundary/baseline.py`: grayscale + blur -> Canny + dilate -> largest 4-point
  contour via `approxPolyDP` -> 4-point perspective warp -> adaptive threshold.
  Runnable as `python -m camscan.pipeline <image> [--debug-dir DIR]`.
- Tested on `clean_09.jpeg` (printed page, plain wood table): works correctly end to
  end — found the true page boundary, warped it flat, produced a clean readable
  black-and-white scan.
- Tested on `clean_01.jpeg` (book on dark table): baseline picked the wrong contour —
  it locked onto the white barcode sticker on the book cover instead of the book's own
  edge, because "largest closed 4-point contour" doesn't distinguish a small
  high-contrast rectangle from the actual document. This is the textbook baseline
  failure mode the aspect-ratio and contrast-scoring improvements are meant to fix, not
  a pipeline bug — confirmed by the debug contour overlay saved to `data/results/debug/`.

Next: run the baseline across the full 38-photo test set to get a per-bucket success
rate, then implement Improvement 1 (aspect-ratio-constrained candidates) and
Improvement 2 (inside/outside contrast scoring, per Zhukovsky et al.).

## 2026-07-26 — Hr ~3.5: Improvement 1 + batch comparison across all 38 photos

- Refactored candidate generation into `camscan/boundary/candidates.py`
  (`quad_candidates`, shared by both methods) so `baseline.py` and the new
  `aspect_ratio.py` aren't duplicating contour/approxPolyDP logic.
- Implemented `aspect_ratio.py`: among the largest candidate quads, picks the one whose
  long/short side ratio is closest to A4 (1.414) or Letter (1.294), rejecting (falling
  back) if nothing is within tolerance (0.35) rather than confidently returning a
  wrong-shaped quad.
- Wrote `compare.py`: runs every registered method over all 38 photos in `data/raw/`,
  scores "success" as (found a non-fallback quad) AND (covers 10-98% of the frame area),
  saves per-image debug overlays to `data/results/debug/<method>/<condition>/`, and
  dumps raw results to `data/results/comparison.json`.
- Bug found + fixed: `quad_candidates` could return a near-zero-area degenerate quad
  that still passed `isContourConvex` (approxPolyDP collapsing a sliver contour into 4
  points) -- added a `min_area_ratio` floor at the source.
- Results after the fix:

  | method | clean | cluttered | low_light | skewed | overall (raw) | overall (corrected) |
  |---|---|---|---|---|---|---|
  | baseline | 9/14 | 0/8 | 1/9 | 1/7 | 11/38 | 10/38 |
  | aspect_ratio | 8/14 | 0/8 | 1/9 | 1/7 | 10/38 | 10/38 |

- Important caveat, found by visually auditing the debug overlays rather than trusting
  the score: aspect_ratio's one "loss" vs baseline (`clean_02`) is actually a case where
  baseline locks onto the wrong region -- a small illustration graphic on the book
  cover, not the book's real boundary -- and the proxy metric can't distinguish "found
  a quad" from "found the *right* quad" without ground-truth corners. aspect_ratio
  correctly rejected that same bad candidate (it isn't A4/Letter-shaped) and fell back
  instead of confidently returning a wrong crop, which the metric doesn't reward. So the
  raw score modestly *understates* aspect_ratio's real value. Manually correcting this
  one known false positive (verified by eye against `data/results/debug/baseline/clean/
  clean_02_contour.png`) puts baseline and aspect_ratio at a tied 10/38 -- not baseline
  ahead. Only this single case has been manually re-verified; the rest of the raw counts
  still carry the same right-region-vs-wrong-region blind spot and haven't been audited
  one by one.
- `cluttered` (0/8) and `low_light` (1/9) are essentially unsolved by both classical
  shape-only methods -- expected, and exactly what Improvement 2 (inside/outside
  contrast scoring) targets next, since it scores by boundary contrast rather than
  shape/size and should behave differently under clutter.

Next: implement Improvement 2 (contrast scoring, per Zhukovsky et al. 2020), add it to
compare.py, re-run the full comparison.

## 2026-07-26 — Hr ~5.5: Improvement 2, a real candidate-generation bug fix, and a full
manual audit that overturns the automated numbers

**Improvement 2.** Implemented `contrast_score.py`: scores each candidate quad by the
color difference between a thin band just inside its border and a thin band just
outside (per Zhukovsky et al., 2020), instead of trusting area or aspect ratio alone.

**The user pointed at `clean_02_contour.png` and said the Dune book was never captured
correctly.** Investigating that directly uncovered a real, foundational bug, not just a
scoring problem:

- The raw Canny edge map barely showed the book's outline at all against the dark
  table -- fixed thresholds (75/200) are tuned for high-contrast scenes and miss a dark
  object on a dark background entirely. Fixed by adding CLAHE local-contrast
  enhancement before blur (`preprocess.py`) and switching Canny to per-image adaptive
  thresholds based on the image's own median intensity (`edges.py`), instead of one
  global threshold.
- Even after that, the book's true contour was being traced but approximated to 11
  noisy, non-convex points -- never reducible to a clean quad. Fixed by taking the
  convex hull of each contour before `approxPolyDP` (`candidates.py`), which strips
  that noise out.
- A related case (`clean_02`, front cover) still failed: the hull-cleaned contour
  approximated to 5 points, one over the cutoff, and got silently discarded. Fixed with
  a progressive-epsilon search (0.02 up to 0.10) that keeps relaxing until it collapses
  to exactly 4 points, instead of only trying one fixed epsilon.
- `contrast_score` needed one more fix on top of all that: even with the real book
  candidate available and much larger, its raw contrast (~20-30, a dark book against a
  dark table) still lost to a small, sharply-contrasted sticker or text block (~130-200)
  in the same frame. Weighting the score by candidate area (linear) fixed the clean
  cases. Pushed the area exponent to 1.5 to try to also fix some low_light/skewed
  cases -- it fixed those but broke others that linear weighting had gotten right
  (25/38 vs 26/38 automated). Reverted to linear. This is now documented as a real,
  accepted limitation of scoring by raw region-mean contrast under uneven lighting, not
  an unsolved tuning problem -- pure contrast alone isn't a reliable signal when the
  true boundary's contrast is inherently weak and inconsistent.

**Then a second, more serious problem showed up:** after those candidate-generation
fixes, `compare.py`'s automated numbers jumped enormously (baseline 11/38 -> 36/38) --
too good to be true for a 2-day classical CV pipeline. Spot-checking baseline's
`cluttered` and `low_light` "successes" against their debug overlays confirmed it:
several were boxes that traced the *entire* stack of overlapping documents instead of
one document, or wildly overshot the actual page into the background curtain, while
still passing the automated proxy check (found a non-fallback quad, 10-98% of frame
area). The proxy metric literally cannot tell "found a quad" from "found the *right*
quad" -- this is the same blind spot flagged after `clean_02` earlier, just at much
larger scale than one image.

**Response: full manual visual audit**, not another automated metric. Built
`scripts/contact_sheet.py` to tile each method/condition's debug overlays into one
grid image (saved to `docs/contact_sheets/`), then visually verified every image each
method claimed as a success against its actual overlay.

| method | clean | cluttered | low_light | skewed | **real total** | (automated total) |
|---|---|---|---|---|---|---|
| baseline | 11/14 | 5/8 | 2/9 | 3/7 | **21/38** | (36/38) |
| aspect_ratio | 12/14 | 4/8 | 4/9 | 2/7 | **22/38** | (32/38) |
| contrast_score | 11/14 | 5/8 | 2/9 | 0/7 | **18/38** | (26/38) |

Findings from the audit itself:
- The three methods are much closer together than the automated score suggested.
  `aspect_ratio` narrowly leads overall, which matches the intended progression
  (baseline -> constrained -> re-scored), though the margin is modest, not dramatic.
- `contrast_score` is genuinely weak on `skewed` (0/7 in manual review) -- raw
  region-mean contrast gets overwhelmed by the sky-brightness gradient across those
  handheld-against-a-window photos, which creates a stronger "edge" signal than the
  document's own boundary.
- All three methods score almost identically on `cluttered` (5/8, 4/8, 5/8) because
  they all re-rank the *same* candidate list -- none of them can tell "this is one
  document" from "this is a stack of documents," so when the true single-document
  candidate isn't clearly dominant, all three converge on the same wrong answer. That's
  a structural gap in this whole candidate-generation-then-rerank approach, not
  something any of the three scoring functions individually can fix -- worth keeping in
  mind for the writeup as the actual boundary of what classical re-ranking can achieve.
- The automated `compare.py` proxy metric is kept in the repo (still useful as a fast
  smoke test / regression check) but is now documented as unreliable for reporting
  real accuracy -- the manual audit numbers above are the ones that should be quoted.

Next: decide whether to pursue the SAM stretch goal (parked pending Jetson-specific
PyTorch wheels) given the cluttered-bucket ceiling just found, or spend remaining time
tightening the classical methods (e.g. a distinct-single-object prior) instead.

## 2026-07-26 — Hr ~7: SAM stretch goal (MobileSAM, box prompt), and a fourth honestly
audited data point

**Setup.** This machine is a Jetson Orin Nano (JetPack 6.2 / L4T R36.4.7, CUDA 12.6).
Installed PyTorch 2.8.0 / torchvision 0.23.0 from the Jetson AI Lab wheel index
(`pypi.jetson-ai-lab.io/jp6/cu126` -- note `.io`, not the commonly-referenced `.dev`
domain, which doesn't resolve). Had to downgrade `numpy<2` to match the wheel's ABI.

**Used MobileSAM instead of full SAM**, disclosed up front rather than swapped
silently: with normal desktop apps open, this machine had ~1GB RAM free (already 2.4GB
into swap). Confirmed this wasn't a hypothetical concern -- full-size loading even
MobileSAM's encoder onto the GPU hit a hard CUDA allocator OOM
(`NVML.../CUDACachingAllocator` assert, `NvMapMemAllocInternalTagged error 12`). Fixed
by forcing CPU inference (`sam_boundary.py`), which works but costs ~15s/image on this
hardware -- documented as a real hardware constraint, not tuned away.

**Two real bugs found and fixed while getting a correct result on `clean_01`:**
1. With a box prompt covering most of the frame, MobileSAM's *highest-scoring* mask
   was sometimes the background frame *around* the document (a concave "C"-shaped
   region hugging 3 sides), not the document itself -- and it out-scored the correct
   mask. A first fix attempt (mask pixel count / its own contour's enclosed area) didn't
   catch this, because a "C" shape has no actual topological hole -- it's simply
   connected, so that ratio comes out near 1.0 regardless. Fixed by comparing mask
   pixel count against the area of the mask's own *minimum bounding rotated rectangle*
   instead (`_rect_fill_ratio`): ~0.99 for a real document, ~0.5 for the C-shaped
   frame. Candidates are filtered by this before ranking by SAM's own score.
2. `mobile_sam` needed `timm` (undeclared transitive dependency for its TinyViT
   backbone) -- not in its package metadata, only discovered at import time.

**Full 38-photo, 4-method comparison, manually audited the same way as before**
(every claimed success checked against its actual overlay in `docs/contact_sheets/`,
not the automated proxy count):

| method | clean | cluttered | low_light | skewed | **real total** |
|---|---|---|---|---|---|
| baseline | 11/14 | 5/8 | 2/9 | 3/7 | 21/38 |
| aspect_ratio | 12/14 | 4/8 | 4/9 | 2/7 | 22/38 |
| contrast_score | 11/14 | 5/8 | 2/9 | 0/7 | 18/38 |
| sam | 12/14 | 3/8 | **5/9** | 3/7 | **23/38** |

Findings:
- SAM comes out narrowly ahead overall (23/38), and the win is concentrated exactly
  where expected: `low_light` (5/9, vs. 2-4/9 for every classical method). SAM's
  learned features recognize "document-shaped object" semantically and don't depend on
  Canny gradients the way all three classical methods do, so weak lighting contrast
  -- which cripples Canny -- doesn't cripple SAM the same way.
- SAM is *worse* than baseline/aspect_ratio on `cluttered` (3/8 vs. 4-5/8). It doesn't
  escape the "one document vs. a stack of documents" problem either -- `cluttered_08`
  (notebook + underlying pages + a ruler) got the same whole-stack answer from SAM as
  from the classical methods. It does fail more *honestly* in some cases though
  (`cluttered_06` correctly triggered a fallback instead of confidently guessing wrong,
  where baseline picked a bad quad) -- not reflected in the raw success count, same
  caveat as the `clean_02` case from Improvement 1.
- No method dominates outright. Each has a specific, explainable strength/weakness by
  condition rather than a clean ranking -- classical contrast/shape scoring vs. learned
  segmentation trade off differently depending on what's actually breaking (weak
  lighting vs. ambiguous object boundaries). This is the real content of the
  comparison, not a single "best method" verdict.

Remaining known gap: none of the four methods reliably solves `cluttered` when the
target document isn't visually dominant against other paper in frame -- that's a
finding about the limits of both approaches at this scope, not an unfixed bug.

## 2026-07-26 — Hr ~8: Tried to improve SAM further, reverted after a full audit
Asked whether SAM could be made more robust/accurate. Diagnosed the three known
failures (`skewed_01`, `low_light_01`, `cluttered_08`) as a box-prompt problem: the
generic box covers ~90% of the frame regardless of where the document actually is, and
when the document isn't centered or isn't the most visually dominant thing in that huge
box, SAM's decoder guesses wrong.

**Tried:** a second, tighter box prompt seeded from the classical candidate generator's
rough localization (`_seed_box`), pooling both boxes' candidate masks and picking the
overall highest-scoring one across both. Cheap to add -- SAM's image encoding is the
expensive part and only runs once per image; a second box only costs another decoder
pass (~+1-7s, confirmed by timing).

**Result: net negative, reverted.** Spot-tested against the 4 known failures first:
fixed one (`skewed_02`, previously only grabbing half the page, though still imperfect
after the fix) and didn't move the other three. But a full audit of the resulting
38-photo run found real regressions the targeted spot-test missed: `clean_01`/`clean_02`
now failed outright (full fallback, previously correct), and `cluttered_06`/
`low_light_06` flipped from honest fallbacks into confidently wrong answers. Root cause:
SAM's own confidence score isn't comparable *across different box prompts* -- for
`clean_01`, the seed box ended up nearly as large as the generic box (padding an
already-large classical candidate by 20% covers almost the whole frame anyway), and
produced a looser, less accurate mask that scored *higher* (0.971) than the correct
mask from the original box (0.877). Picking the global max-by-score across both boxes
therefore sometimes picked the worse answer. Reverted to the single-box version;
confirmed the revert reproduces the original run's automated numbers exactly
(clean 12/14, cluttered 4/8, low_light 6/9, skewed 5/7, 27/38 raw), so the previously
audited 23/38 real total still stands unchanged.

Takeaway for anyone trying to push this further: SAM's reported mask score is not a
reliable arbiter for comparing across prompts/scales, any more than the classical
contrast score was reliable across candidate sizes (same shape of problem as the
contrast_score area-weighting issue in Improvement 2). A real fix would need either
ground-truth-supervised calibration of that score, or a different prompting strategy
entirely (e.g. automatic whole-image mask generation instead of box prompts) -- ruled
out earlier as too slow for CPU-only inference on this hardware. Treating 23/38 as the
final, honestly-audited SAM result for this project's scope.

## 2026-07-26 — Hr ~9: YOLO26-pose corner detector tested, hit a real dataset bug,
pivoted to a YOLO+classical hybrid

**Setup.** I had separately trained a YOLO26n-pose model (`model/best(3).pt`,
1-class "corners", `kpt_shape=[4,3]`, 640x640, 288 epochs, reported 0.994 mAP50) outside
this repo and wanted it tested as a fifth boundary-detection method, same as SAM. Added
`ultralytics==8.4.106` (pip; not previously installed here), wrote
`camscan/boundary/yolo_pose.py` following the same lazy-load/shared-signature pattern as
`sam_boundary.py`, wired into `pipeline.py`/`compare.py`. Forced to CPU for the same
reason as MobileSAM -- this Jetson OOMs the CUDA allocator under normal desktop load.
Fed the model the original full-resolution photo (not the shared 500px detection frame)
since it was trained at 640x640 and Ultralytics letterboxes/rescales internally anyway;
its output quad is scaled back into the shared 500px coordinate space so it stays
comparable to every other method's score.

**Bug found: every prediction's 4 keypoints collapse into 2 distinct points**
(keypoint 0 ~= keypoint 1, keypoint 2 ~= keypoint 3, separation ~0.1-0.5% of the box
diagonal -- noise level), across every test photo in every condition, with no
exceptions. This produced a systematic 0/38 automated score for `yolo_pose`: quads
built from 2 real points (repeated) are degenerate/near-zero-area and always fail the
area-ratio check.

Investigated thoroughly before concluding anything, since the I pushed
back on an initial wrong read:
- First guess (wrong, retracted after I pushedback): Claude misread `model/train_batch0.jpg`
  at thumbnail resolution as having only 2 annotated keypoints per document. At full
  zoom the training visualization does show what look like 4 distinct dot colors across
  different cells.
- Checked for an ultralytics version mismatch between training and here -- ruled out,
  exact version match (8.4.106) confirmed via the checkpoint's own saved `version` field.
- Read the actual `Pose26` head source (yolo26's new pose head, with a `RealNVP`
  normalizing-flow submodule not present in the older `Pose` head) suspecting the flow
  model was needed for correct decode and being skipped. Also ruled out: `RealNVP` here
  implements RLE (Residual Log-likelihood Estimation), a training-time uncertainty loss
  only (matches `rle: 1.5` in `model/args.yaml`), not an inference-time coordinate
  transform -- a red herring.
- Decisive test: ran the model directly on crops taken from `model/train_batch0.jpg`
  itself -- images the model was trained on and should reproduce near-perfectly given
  0.994 mAP50. Got the exact same collapse pattern. A real inference/decode bug would be
  expected to disagree with training-time behavior; reproducing it exactly on training
  data itself points at the label/export step instead.
- Followed up with pixel-level zoom on the training visualization at the actual
  documented corners (not thumbnail scale): the bottom-left corner shows one visible
  dot, not two separated ones. Conclusion: the training data/export likely encodes only
  2 distinct corner locations per document (duplicated across the 4 keypoint slots),
  not 4 independent corners, despite `kpt_shape=[4,3]` -- most likely a labeling-tool or
  Roboflow-export quirk from how the source dataset was built, not a bug in this repo's
  code. Not independently confirmed against the original Roboflow project/label files
  (those lived on Colab, `/content/Paper-Corner-Detection-2`, not saved locally) --
  this is the most likely explanation given the evidence, not a certainty.

**Pivot : use YOLO for what its output actually supports.**
The bounding box (built from the 2 real corners the model does predict reliably) is a
normal, undamaged detection-head output, unaffected by the keypoint-collapse issue.
Wrote `camscan/boundary/yolo_hybrid.py`: run YOLO on the full-res image for a coarse
box, crop to it (+8% margin) in the shared 500px coordinate space, then run the same
classical Canny + convex-hull + progressive-epsilon quad search the other methods use,
but *inside* that crop -- where the document is the dominant shape in frame, instead of
competing with a whole tabletop or messy pile. `min_area_ratio` for the inner search is
0.35 (of the crop, not the frame) since a real document should fill most of a
YOLO-localized crop.

**Full 38-photo run, manually audited the same way as every other method** (automated
proxy score is a fast smoke test only, not a real accuracy number -- established back in
Hr ~5.5):

| method | clean | cluttered | low_light | skewed | **real total** | (automated) |
|---|---|---|---|---|---|---|
| baseline | 11/14 | 5/8 | 2/9 | 3/7 | 21/38 | (36/38) |
| aspect_ratio | 12/14 | 4/8 | 4/9 | 2/7 | 22/38 | (32/38) |
| contrast_score | 11/14 | 5/8 | 2/9 | 0/7 | 18/38 | (26/38) |
| sam | 12/14 | 3/8 | 5/9 | 3/7 | 23/38 | (27/38) |
| yolo_pose | 0/14 | 0/8 | 0/9 | 0/7 | **0/38** | (0/38) |
| yolo_hybrid | 12/14 | 4/8 | 2/9 | 4/7 | **22/38** | (27/38) |

(baseline/aspect_ratio/contrast_score/sam rows are the previously audited numbers,
unchanged, shown for reference.)

Findings:
- `yolo_pose` is not a usable data point in its current form -- 0/38 isn't "the model is
  bad," it's "this comparison can't measure a model whose keypoint output is
  structurally broken." Kept in the repo/comparison table for honesty (a silently
  dropped method would be worse), but shouldn't be read as "learned pose regression
  loses to everything else."
- `yolo_hybrid` lands at 22/38, statistically tied with `aspect_ratio` (22/38) and
  `sam` (23/38) -- not a clear winner or loser. Its automated score (27/38) overstated
  it by 5, the same right-region-vs-wrong-region blind spot flagged repeatedly
  throughout this project: several "successes" were quads that sliced diagonally across
  only part of the true page (`clean_01`, `clean_06`, `low_light_03/05/09`,
  `cluttered_08`) while still passing the non-fallback + 10-98%-area-ratio proxy check.
- Where it actually helps: `skewed` (4/7, tied for best alongside baseline's historical
  strength there) -- YOLO's coarse localization reliably finds the book/page even
  against a bright sky background that confuses Canny, and once cropped tightly the
  classical quad search handles the perspective skew fine.
- Where it doesn't help: `low_light` (2/9) -- YOLO's box localization on these backlit
  photos is often fine, but the classical search *inside* the crop still needs a real
  Canny edge to find the document's own boundary, and weak backlit contrast defeats it
  the same way it defeats every from-scratch classical method. Cropping first doesn't
  fix a fundamentally weak edge signal.
- `cluttered` (4/8) is roughly the same story as every other method on this bucket:
  when YOLO's box already spans nearly the whole frame (an open book with two facing
  pages, or a document buried in a stack), cropping to it doesn't isolate a single
  dominant document shape, so the inner classical search is back to the same
  "which one is the real document" ambiguity that's limited every method so far.

**Overall verdict for this stretch goal:** the raw pose-regression approach didn't
produce a usable result on this dataset/model, most likely due to a labeling/export
issue upstream rather than a modeling failure -- flagged rather than silently
worked around. The YOLO+classical hybrid built on top of it is a legitimate fifth
data point, competitive with (not clearly better than) the existing methods, with the
same structural blind spot on ambiguous multi-document clutter that every method in
this project has run into. No single method dominates; the real content of this project
continues to be *which specific condition each method's specific mechanism helps or
hurts*, not a single leaderboard-style "best method."

## 2026-07-27 -- A second training run (YOLOv8n-pose) actually works: direct 4-keypoint detection, no hybrid needed

 I trained a second model om google colab and saved it as attempt in `model/attempt 2_yolov8/` --
same task (4-corner document pose, single class), same dataset lineage
(`Paper-Corner-Detection-2`), but `yolov8n-pose.pt` as the base model instead of
YOLO26n-pose, 50 epochs, `rle: 1.0` (vs the YOLO26 run's `rle: 1.5`). Final-epoch
metrics from `results.csv`: box mAP50-95 ~0.907, pose mAP50-95 ~0.474, pose mAP50
~0.973 -- a real, converged run, not a partial one.

**First check: does this run have the same keypoint-collapse bug as the YOLO26
attempt?** Spot-checked predictions directly (`model.predict(..., conf=0.25)`) on
three raw photos before writing any pipeline code:
- `clean_01`: 4 keypoints at `(951,184)`, `(328,158)`, `(262,683)`, `(936,646)` --
  four genuinely distinct corners spanning the whole detected box.
- `clean_02`, `cluttered_01`: same pattern, four distinct corners forming a real quad
  each time.

Confirms my claim directly: this checkpoint does not have the 2-point
collapse that made `yolo_pose.py` (the YOLO26 attempt) score 0/38. One caveat found
during this check: per-keypoint confidence on this model is noisy -- one otherwise
geometrically correct corner came back with confidence ~0.005. Gating on a per-point
confidence floor (the same `min_kpt_conf=0.5` pattern used in `yolo_pose.py`) would
have thrown away good detections, so `yolo_v8_pose.py` only filters on overall box
confidence and trusts all 4 returned points once a detection passes that bar.

**Environment note (unrelated to the model, but blocked the run):** system `numpy`
had drifted to 2.2.6 at some point after the last session, which broke `torch` import
entirely (this Jetson's nv24.08 torch wheel is compiled against numpy 1.x) -- not
just for this new model, for every learned method. Fixed with `pip install
"numpy<2"` (landed on 1.26.4). Also found `imutils` and `mobile_sam` missing from the
user-site packages (present in a previous session, absent now -- most likely a
system-level package cleanup between sessions, not caused by anything in this repo).
Reinstalled `imutils` since `camscan/preprocess.py` depends on it directly.
`mobile_sam` was left alone (out of scope for this task) and `sam` was excluded from
this run; **`yolo_pose.py`'s hardcoded checkpoint path
(`model/best(3).pt`) is now stale** since the model folders were reorganized into
`attempt 1_yolo26/` and `attempt 2_yolov8/` -- `yolo_pose` and `yolo_hybrid` were
excluded from this comparison run for that reason and still need that path fixed
before they can run again.

**Built `camscan/boundary/yolo_v8_pose.py`:** same interface and full-res-input
convention as `yolo_pose.py` (checkpoint at
`model/attempt 2_yolov8/weights/best.pt`, predicts at 640x640 on the original image,
scales the resulting quad back into the shared 500px detection space), but simpler --
no per-keypoint confidence gate (see above), and since the 4 points are directly
usable this is a straight `_order_corners` + area-sanity-check, no classical
refinement layer needed.

**Automated comparison (baseline / aspect_ratio / contrast_score / yolo_v8_pose only,
`sam`/`yolo_pose`/`yolo_hybrid` excluded per the environment notes above):**

```
method            clean  cluttered  low_light  skewed  overall
baseline          14/14     8/8        9/9      6/7     36/38
aspect_ratio      13/14     7/8        7/9      5/7     32/38
contrast_score    12/14     8/8        4/9      2/7     26/38
yolo_v8_pose      14/14     7/8        6/9      5/7     32/38
```

**Manually audited** against the debug overlays (per this project's standing rule
that the automated proxy can't distinguish "a quad" from "the right quad"):

| condition | automated | audited |
|---|---|---|
| clean | 14/14 | 14/14 |
| cluttered | 7/8 | 4/8 |
| low_light | 6/9 | 5/9 |
| skewed | 5/7 | 5/7 |
| **total** | **32/38** | **28/38** |

- `clean`: all 14 genuinely correct, including skewed book-cover shots (`clean_01`,
  `clean_02`, `clean_05`, `clean_06`) and curled/creased pages (`clean_13`,
  `clean_14`) -- the model handles real perspective distortion on the corners
  correctly, which is exactly the property that the dataset was labeled for.
- `cluttered`: the automated score (7/8) overstated it -- 3 of the 7 "successes"
  were quads that clearly overshot the true page boundary onto background clutter
  once viewed at full resolution (`cluttered_01`: top edge cuts across a background
  page; `cluttered_04`: overshoots onto a background sheet past the true bottom-left
  corner; `cluttered_08`: overshoots right onto the ruler/background past the
  notebook's real edge). Real correct count: `cluttered_03`, `05`, `06`, `07` -- 4/8.
  `cluttered_02` is a genuine fallback (no detection).
- `low_light`: audited close to the automated score. Real correct: `low_light_03`,
  `04`, `07`, `08`, `09` -- all five hold a tight, accurate quad on the page despite
  harsh backlighting from a window. `01`, `02`, `05`, `06` are genuine fallbacks (no
  detection at all) -- backlit silhouette photos where the model doesn't fire a
  confident box, not a wrong-quad problem.
- `skewed`: all 5 non-fallback detections audited as correct, including two
  photos where the page is held up against a bright, textured sky/skyline
  background that has defeated Canny-based methods throughout this project
  (`skewed_04`, `skewed_06`) and one where two book pages/covers touch and the model
  still separates the correct one (`skewed_02`, `skewed_06`). `skewed_01` and
  `skewed_03` are fallbacks.

**Verdict:** this is the best audited score of any method tested in this project so
far (28/38, vs the previous best of `sam` at 23/38 and `baseline` at 21/38), and the
first learned method that actually works as originally intended -- direct 4-corner
regression, no classical fallback layer required. Its failure mode is narrower and
more honest than the classical methods': when it's wrong, it's usually a clean
fallback (no detection) rather than a confidently-wrong quad, except on `cluttered`
where 3 of 8 detections are confidently wrong in the same "found a real quad, wrong
region" way every other method struggles with. `low_light` fallbacks and `cluttered`
overshoot remain the two weakest spots; unlike `yolo_hybrid`, there's no classical
refinement step to add here since the model's own corners are already the intended
final answer.

**Still open :** `yolo_pose.py`'s checkpoint path is stale
after the `model/` reorg and needs updating to
`model/attempt 1_yolo26/best(3).pt` before that method (or `yolo_hybrid`, which
depends on it) can run again. `mobile_sam` is missing from the environment and `sam`
could not be run this session.

## 2026-07-27 — Frontend built, attempt 5 (YOLO26s-pose, 800px) trained and swapped in as
production, fallback architecture reworked, and full ONNX conversion

I built a Next.js/FastAPI frontend (`frontend/`, `api/main.py`) around the pipeline in
this same stretch: detect → manual quad correction → warp/enhance → export, backed by
an in-memory per-session store (`api/main.py`'s `_SESSIONS`, 30-minute TTL, no
database — deliberate for a single-instance deployment, see the ONNX/deployment
section below for the scaling tradeoff this implies).

**Attempt 3 (`model/attempt 3_yolo26/`):** another YOLO26n-pose run, same dataset
lineage as attempt 1, imgsz 640. Superseded by attempt 4 and never wired into the
pipeline (`yolo26_v2_pose.py` existed but was never added to `ALL_METHODS`) — I removed
it during the 2026-07-27 cleanup pass below as genuinely dead code, not just unused.

**Attempt 4 (`model/attempt 4_yolo26/`):** a cleaner YOLO26n-pose run than attempt 1
on the same 640px/50-epoch recipe, but on a different dataset than attempts 1-3: I
trained this one on 3,000 training images, 200 validation, 200 test. Wired in as
`yolo26_doccorner_pose.py` and made `DEFAULT_METHOD`. Became my production model ahead
of this entry (best audited score at the time) — full attempt-4 numbers were
superseded by attempt 5 below before I wrote a dedicated log entry for attempt 4 in
isolation.

**Attempt 5 (`model/attempt 5_yolo26s/`):** a larger-backbone run — `yolo26s-pose.pt`
instead of attempt 4's `yolo26n-pose.pt`, 80 epochs (vs 50), trained at **imgsz 800**
(vs attempt 4's 640 — this model must be predicted at 800, not 640, or accuracy
degrades). Same dataset lineage as attempt 4, but scaled up significantly: 10,000
training images, 1,000 validation, 1,000 test (vs attempt 4's 3,000/200/200), with
better fine-tuning on top. Head-to-head against attempt 4 on my project's 38-image
test set (results saved to `data/results/attempt4_yolo26/` and
`data/results/attempt5_yolo26s/`): near-identical raw detection rate (27/38 vs attempt
4's 28/38) but visibly tighter/cleaner quads on shared hits, especially on skewed and
cluttered images. I chose this one over attempt 4 for the quad-quality edge, not
because it wins on raw recall — I repointed `yolo26_doccorner_pose.py` at attempt 5
and `DEFAULT_METHOD` follows.

**I reworked the fallback architecture from "first non-None wins" to "score-and-pick":**
the classical `FALLBACK_CHAIN` (`baseline`, `aspect_ratio`, `contrast_score`) used to
return whichever method ran first and found *any* quad, even when a later method in
the same chain had found a clearly better one for the same image (concrete example:
`clean_02`, where `baseline`'s quad clips the document but `contrast_score`'s
doesn't — `baseline` used to win purely by running first). I fixed this by extracting
a public `score_quad` from `contrast_score.py`'s internal scoring function and
changing `pipeline.detect_boundary` to run every method in the chain, then pick the
single best-scoring quad across all of them — see `camscan/pipeline.py` and
`camscan/boundary/contrast_score.py`. I verified this against the existing 38-image
set: same 10/38 fallback-triggering images as before, but 3 of them now get a visibly
better quad instead of the first-found one.

**I exported attempt 5 to ONNX for faster serving:** `model.export(format='onnx',
imgsz=800, opset=12, simplify=True)` → `model/attempt 5_yolo26s/weights/best.onnx`.
I found and fixed two integration issues: (1) ONNX files don't carry Ultralytics' task
metadata the way `.pt` checkpoints do, so loading without `task="pose"` silently
misdetects as `task="detect"` and drops the keypoint head's output entirely — fixed by
loading with `YOLO(path, task="pose")` explicitly; (2) `.to("cpu")` only works on
PyTorch `nn.Module`-backed models and raises `TypeError` on an ONNX-loaded one — fixed
by passing `device="cpu"` directly to `.predict()` instead. I benchmarked this at
~2.4x faster per-image than the `.pt` model on my Jetson's CPU (1899.8ms → 779.6ms).
I verified it was lossless: a full 38-image regression run against the ONNX-based
pipeline produced the exact same 12/38 fallback list as attempt 5's original `.pt`
weights — this is a serving-speed change, not a detection-quality change.
`camscan/boundary/yolo26_doccorner_pose.py` now loads
`model/attempt 5_yolo26s/weights/best.onnx` at `PREDICT_IMGSZ=800`.

**Repo cleanup:** I removed `yolo_pose.py` and `yolo_hybrid.py` (broken since the
`model/` reorg — stale checkpoint path, documented 0/38 score, never fixed), the
never-wired `yolo26_v2_pose.py` (attempt 3), and `yolo_v8_pose.py` (attempt 2 — a
real, working comparison method at the time, but dropped along with `model/attempt
1-4` in favor of standardizing the repo on attempt 5 as my single production model).
`model/` now holds only attempt 5's `args.yaml`, `results.csv`/`results.png` (training
report), and `weights/best.onnx` under git — I deleted intermediate epoch checkpoints
(`epoch0/10/20/30.pt`, `last.pt`) and per-run training visualizations
(label/batch/PR-curve/confusion-matrix images) as regenerable training scratch, and
`.gitignore` now excludes `model/*/*.png|jpg|jpeg` except `results.png`. `.pt` weights
stay gitignored as before (`best.pt` is kept locally for retraining/reference, not
committed). I trimmed `data/results/` to the outputs my current tooling (`compare.py`,
`scripts/contact_sheet.py`) actually regenerates, plus the attempt4-vs-attempt5
comparison; I removed stale outputs referencing deleted methods (`yolo_pose`,
`yolo_hybrid`, `yolo_v8_pose`, `yolo26_v2_pose`) and superseded duplicate
preview/contact-sheet directories from earlier ad-hoc runs.

**Still open:** `onnxruntime` is only installed in my local dev venv, not yet added
to `requirements.txt` — I need to fix this before a fresh deploy (e.g. Hugging Face
Spaces) can actually load the ONNX model.

## 2026-07-27 — Deployed to Railway + Vercel, dropped ultralytics from the served
path, added a real test suite

`onnxruntime`/`ultralytics` added to `requirements.txt` (fixing the open item above).
Deployment target changed from the originally-discussed Hugging Face Spaces to
**Railway** (backend) + **Vercel** (frontend) after checking HF's actual pricing:
Docker Spaces require a paid PRO plan just to *create*, even though the underlying
CPU hardware is free — not worth it for this project's current stage. Railway's free
tier is thin ($1/mo credit after an initial trial) but its $5/mo Hobby tier is cheap
and, unlike Fly.io (no real free tier left for new accounts in 2026) or Render (works,
but sleeps after 15 min idle on its free tier), doesn't need a credit card to start.

**Dropped `ultralytics` from the deployed image.** The ONNX-loading path
(`yolo26_doccorner_pose.py`) used `ultralytics.YOLO(path, task="pose")` as a thin
wrapper around ONNX Runtime — convenient, but `ultralytics` hard-depends on
`torch`+`torchvision` (~4GB) regardless of whether PyTorch itself is ever used for
inference, which it isn't here. Built a Docker image to check the actual cost: **5.92GB**,
almost entirely torch/torchvision. Rewrote `_run_model` to call
`onnxruntime.InferenceSession` directly, reimplementing what the wrapper did
internally:
- `_letterbox`: aspect-preserving resize + 114-gray pad to 800x800, matching
  ultralytics' own default preprocessing. Verified by direct comparison — ran the
  same image through both the `ultralytics`-wrapped path and the raw-ONNX path,
  confirmed matching confidence (`0.850814` vs `0.85081`) and box/keypoint
  coordinates to 3+ decimal places before trusting the rewrite.
- Manual postprocessing of the exported graph's `(1, 300, 18)` output (NMS already
  baked in by the export step): box + confidence + class + 4 keypoints per row,
  unletterboxed back into the original image's coordinate space.
- Full 38-image regression via `compare.py` after the rewrite reproduced the exact
  same per-condition scores as the `ultralytics`-backed version — confirms the
  rewrite is lossless, not just "looks right on one image."

Combined with switching `opencv-python` → `opencv-python-headless` (no GUI/X11 bindings
needed in a container) and splitting a deploy-only `requirements-deploy.txt` from the
full dev `requirements.txt`, the image dropped to **679MB** — an 8.7x reduction from
5.92GB.

**Railway deploy troubleshooting:** the first deploy attempt sat at "Queued" indefinitely
because Railway's dashboard had silently picked its own auto-builder (Railpack) instead
of the repo's `Dockerfile` — fixed by explicitly setting Builder to `Dockerfile` and the
Dockerfile path to `Dockerfile` (blank/no leading slash) in Settings → Build. A second
issue after that: the container came up healthy (confirmed via deploy logs showing
`Uvicorn running on http://0.0.0.0:8080` and a successful internal healthcheck `200 OK`)
but was unreachable from outside — Settings → Networking → Edit Port was unset, so
Railway's edge proxy had nothing to route the public domain to; setting it to 8080 (the
port `$PORT` actually resolves to in this environment) fixed external routing. Also set
Healthcheck Path to `/api/health` so Railway can verify liveness using the endpoint this
project already exposes for that purpose, rather than leaving it unset.

**Repo cleanup, round 2:** deleted `example/CamScanner-UT.ipynb` — a pre-project
scaffolding notebook (fill-in-the-hints tutorial format, referencing a nonexistent
`images/example.jpg`) left over from before `camscan/` existed, unreferenced by
anything.

**Added `tests/` (pytest, 15 tests):** targeted at the logic most likely to silently
regress rather than broad coverage --
- `test_pipeline_fallback.py`: the score-and-pick fallback behavior, including a
  regression test that reproduces the exact bug the score-and-pick rework fixed
  (asserts the higher-scoring quad wins even when a worse one runs first in
  `FALLBACK_CHAIN`) via monkeypatching `pipeline.METHODS` and `pipeline.score_quad`
  directly rather than depending on real image content.
- `test_yolo26_doccorner_pose.py`: `_order_corners` (including a rotated,
  non-axis-aligned quad, not just the trivial already-ordered case), `_box_iou`
  (identical/disjoint/half-overlap/degenerate-zero-area boxes), and `_letterbox`
  (square/wide/tall images, asserting scale and padding placement match the
  aspect-preserving contract) -- all pure-math, no model file or ONNX Runtime needed.
- `test_pipeline_smoke.py`: one real end-to-end `scan()` call, catching
  "the whole pipeline is broken" regressions cheaply without re-deriving
  `compare.py`'s detection-accuracy scoring.

Classical CV methods' exact pixel outputs were deliberately left untested at the unit
level — `compare.py`'s scored regression suite already covers those, and they're
inherently visual/fuzzy in a way unit tests don't suit well.
