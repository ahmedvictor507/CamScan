import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from camscan.preprocess import resize_for_detection, to_blurred_gray
from camscan.edges import detect_edges
from camscan.boundary.candidates import fallback_frame_contour
from camscan.boundary import baseline, aspect_ratio, contrast_score
from camscan.boundary.contrast_score import score_quad
from camscan.warp import four_point_transform, deskew, detect_orientation, rotate_by
from camscan import enhance
from camscan.enhance import enhance_scan

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "tiff", "webp", "pdf"}

METHODS = {
    "baseline": baseline.find_document_contour,
    "aspect_ratio": aspect_ratio.find_document_contour,
    "contrast_score": contrast_score.find_document_contour,
}
ALL_METHODS = list(METHODS) + ["sam", "yolo_pose", "yolo_hybrid", "yolo_v8_pose", "yolo26_doccorner_pose"]  # resolved lazily -- see _resolve_method
_NEEDS_ORIGINAL_IMAGE = {"yolo_pose", "yolo_hybrid", "yolo_v8_pose", "yolo26_doccorner_pose"}
DEFAULT_METHOD = "yolo26_doccorner_pose"  # attempt-5 (ONNX): best audited score so far, see docs/progress_log.md


def _resolve_method(method):
    if method == "sam":
        # deferred: importing torch/mobile_sam costs real time and memory, not worth
        # paying on every run of the (fast, lightweight) classical methods
        from camscan.boundary import sam_boundary
        return sam_boundary.find_document_contour
    if method == "yolo_pose":
        from camscan.boundary import yolo_pose
        return yolo_pose.find_document_contour
    if method == "yolo_hybrid":
        from camscan.boundary import yolo_hybrid
        return yolo_hybrid.find_document_contour
    if method == "yolo_v8_pose":
        from camscan.boundary import yolo_v8_pose
        return yolo_v8_pose.find_document_contour
    if method == "yolo26_doccorner_pose":
        from camscan.boundary import yolo26_doccorner_pose
        return yolo26_doccorner_pose.find_document_contour
    return METHODS[method]


# Tried if `method` returns nothing -- a classical method's "wrong" quad is a much
# better manual-edit starting point for a user than the full-frame fallback rect, so
# exhaust these before giving up entirely. Unlike a simple ordered chain, every method
# here is run and the results are scored against each other (see detect_boundary) --
# an ordered "first non-None wins" chain was found to pick a badly-fit quad from an
# earlier method even when a later method in the same chain had a much better one for
# the same image (e.g. clean_02: baseline's quad clips the document, contrast_score's
# doesn't) -- see docs/progress_log.md.
FALLBACK_CHAIN = ("baseline", "aspect_ratio", "contrast_score")


def detect_boundary(resized_image, method="baseline", original_image=None):
    """Returns (quad, used_fallback) in the resized image's coordinate space.

    `original_image` is optional and only used by methods that do better fed their
    native training resolution instead of the shared 500px-wide detection frame every
    classical/SAM method uses (yolo_pose, and yolo_hybrid's YOLO localization step).

    If `method` finds nothing, every method in FALLBACK_CHAIN (skipping `method` itself
    if it's already in the chain) is run and scored by boundary contrast (see
    contrast_score.score_quad), and the single best-scoring quad across all of them is
    kept -- not just whichever method happens to run first and return non-None. Falls
    back to the full-frame rect only if every method in the chain returns nothing at
    all: a classical method's best-guess quad is a far better starting point for a user
    to manually correct than "no crop at all", even if it isn't accurate enough to trust
    unedited.
    """
    gray = to_blurred_gray(resized_image)
    edge_map = detect_edges(gray)

    def run(m):
        if m in _NEEDS_ORIGINAL_IMAGE:
            return _resolve_method(m)(edge_map, resized_image, detection_image=original_image)
        return _resolve_method(m)(edge_map, resized_image)

    quad = run(method)
    if quad is not None:
        return quad, False

    candidates = []
    for fallback_method in FALLBACK_CHAIN:
        if fallback_method == method:
            continue
        quad = run(fallback_method)
        if quad is not None:
            candidates.append(quad)

    if candidates:
        best = max(candidates, key=lambda q: score_quad(q, resized_image))
        return best, True

    return fallback_frame_contour(resized_image.shape), True


def detect(image_path, method=DEFAULT_METHOD):
    """Detection-only step: loads the photo and returns (original, resized, ratio,
    quad, used_fallback). Split out from warp/enhance so a caller (e.g. a future UI)
    can show the detected quad to a user and accept manual corrections to it before
    calling warp_and_enhance -- `used_fallback=True` is the signal that the returned
    quad is a lower-confidence guess worth surfacing for review, not a final answer.
    """
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(image_path)

    resized, ratio = resize_for_detection(original)
    quad, used_fallback = detect_boundary(resized, method=method, original_image=original)
    return original, resized, ratio, quad, used_fallback


def detect_batch(image_path):
    """Batch variant of detect(): returns (original, resized, ratio, quads) where
    `quads` is a list of zero or more document quads found in one photo, instead of
    detect()'s single best quad. Only meaningful for yolo26_doccorner_pose -- the
    classical fallback methods (baseline/aspect_ratio/contrast_score) each return a
    single best-guess quad by construction, so there's no multi-document fallback here;
    an empty `quads` list means nothing was confidently found, same spirit as detect()'s
    fallback-to-full-frame but without a sensible multi-document equivalent to fall
    back to. See yolo26_doccorner_pose.find_document_contours' docstring: this path is
    unverified against real multi-document photos so far.
    """
    from camscan.boundary import yolo26_doccorner_pose

    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(image_path)

    resized, ratio = resize_for_detection(original)
    quads = yolo26_doccorner_pose.find_document_contours(resized, detection_image=original)
    return original, resized, ratio, quads


def warp_and_enhance(original, quad, ratio, enhance_mode="auto", enhance_strength=enhance.DEFAULT_STRENGTH, manual_rotation=0):
    """Takes a quad in the resized image's coordinate space (either straight from
    detect(), or edited by a user first) and produces the final enhanced scan.

    Orientation is corrected in two stages, coarse before fine: `detect_orientation`
    first fixes a sideways (90/270) photo so its text lines are horizontal, then
    `deskew` cleans up the residual small-angle tilt -- deskew's angle folding assumes
    the page is already close to upright, so running it before the coarse fix would
    misread a 90-degree-rotated page's "horizontal" lines as vertical ones needing a
    huge correction. `manual_rotation` (0/90/180/270) applies on top of that, letting a
    user fix the one case classical geometry can't tell apart from upright -- upside
    down -- since a page and its 180-degree rotation look identical to any line-based
    heuristic; see detect_orientation's docstring.
    """
    warped = four_point_transform(original, quad * ratio)
    warped = rotate_by(warped, detect_orientation(warped))
    warped = deskew(warped)
    warped = rotate_by(warped, manual_rotation)
    return enhance_scan(warped, mode=enhance_mode, strength=enhance_strength)


def scan(image_path, method=DEFAULT_METHOD, enhance_mode="auto", enhance_strength=enhance.DEFAULT_STRENGTH, manual_rotation=0, debug_dir=None):
    """Convenience wrapper for the CLI/simple callers: runs detect() then
    warp_and_enhance() back-to-back with no manual-edit step in between."""
    original, resized, ratio, quad, used_fallback = detect(image_path, method=method)

    if debug_dir is not None:
        overlay = resized.copy()
        cv2.drawContours(overlay, [quad.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(Path(debug_dir) / f"{Path(image_path).stem}_contour.png"), overlay)

    enhanced = warp_and_enhance(original, quad, ratio, enhance_mode=enhance_mode, enhance_strength=enhance_strength, manual_rotation=manual_rotation)
    return enhanced, used_fallback


def save_output(image, output_path):
    """Writes `image` (a cv2 array, either grayscale/binary or BGR color) to
    `output_path`. PNG/JPG/TIFF/WEBP go through cv2 directly; PDF doesn't have cv2
    support, so it's routed through Pillow instead -- a single-page PDF wrapping the
    same pixel data.
    """
    ext = output_path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported output format {ext!r}, choose from {sorted(SUPPORTED_FORMATS)}")

    if ext == "pdf":
        if image.ndim == 2:
            pil_image = Image.fromarray(image).convert("L")
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_path, "PDF")
    else:
        cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser(description="Scan a document photo into a flat, enhanced image.")
    parser.add_argument("image", type=Path, help="Path to the input photo")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <name>_scan.png next to input)")
    parser.add_argument("--method", choices=ALL_METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--enhance", choices=["auto", "gray", "bw", "color"], default="auto", help="Output look: auto-detect (default), soft grayscale, hard black & white, or force color enhancement")
    parser.add_argument("--strength", type=int, default=enhance.DEFAULT_STRENGTH, help="Enhancement intensity 0-100 (default: 50)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Manual rotation on top of auto-orientation/deskew (e.g. to fix upside-down pages)")
    parser.add_argument("--format", choices=sorted(SUPPORTED_FORMATS), default=None, help="Output format (default: inferred from --output's extension, or png)")
    parser.add_argument("--debug-dir", type=Path, default=None, help="If set, save the contour overlay here")
    args = parser.parse_args()

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    result, used_fallback = scan(args.image, method=args.method, enhance_mode=args.enhance, enhance_strength=args.strength, manual_rotation=args.rotate, debug_dir=args.debug_dir)
    if used_fallback:
        print(f"Warning: no document boundary found, used the full frame as a fallback")

    fmt = args.format or (args.output.suffix.lstrip(".").lower() if args.output else "png")
    output_path = args.output or args.image.with_name(f"{args.image.stem}_scan.{fmt}")
    if args.output and args.format and args.output.suffix.lstrip(".").lower() != args.format:
        output_path = args.output.with_suffix(f".{args.format}")

    save_output(result, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
