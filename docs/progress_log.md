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

User asked whether SAM could be made more robust/accurate. Diagnosed the three known
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
