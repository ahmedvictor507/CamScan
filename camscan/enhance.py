import cv2
import numpy as np

# A uniform lighting tint (warm bulb, cool backlight) shifts every pixel's B/G/R by
# roughly the same amount, so it barely moves max-channel minus min-channel per pixel --
# unlike HSV saturation, which conflates "tinted" with "actually colorful" and misreads
# backlit/warm-lit plain paper as a color document. Real color content (book covers,
# printed photos) has channels that diverge much more per pixel, well past what a
# uniform cast alone produces. Threshold picked empirically against this project's test
# photos: plain-paper shots (including backlit/warm-cast ones) land under ~12, real
# covers/photos land at 40+.
COLOR_CHANNEL_SPREAD_THRESHOLD = 25

# strength=0..100 maps linearly onto these ranges. Midpoints (50) reproduce this
# project's hand-tuned defaults from earlier passes; the ranges themselves were picked
# so 0 stays close to a raw lighting-normalized image and 100 revisits the heaviest
# look tried during tuning, without exceeding it.
BW_C_RANGE = (0.0, 12.0)
COLOR_CLIP_RANGE = (0.0, 1.6)
GRAY_CLIP_RANGE = (0.0, 3.5)
DEFAULT_STRENGTH = 50


def _lerp(range_: tuple[float, float], strength: int) -> float:
    lo, hi = range_
    t = max(0, min(100, strength)) / 100.0
    return lo + (hi - lo) * t


def _is_color_document(warped_bgr: np.ndarray) -> bool:
    b, g, r = (warped_bgr[:, :, i].astype(np.int16) for i in range(3))
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    return float(np.median(spread)) > COLOR_CHANNEL_SPREAD_THRESHOLD


def _enhance_gray_page(warped_bgr: np.ndarray, strength: int = DEFAULT_STRENGTH) -> np.ndarray:
    """Scanner-like look that keeps continuous gray tones instead of snapping every
    pixel to pure black/white -- real scanner apps ship this as a distinct "gray mode"
    separate from their hard black & white filter, because thresholding (see
    _enhance_text_page) inherently produces jagged, anti-alias-free text edges no
    choice of threshold algorithm can avoid.

    Illumination is normalized via the "divide trick": a large-kernel blur estimates
    the local background/shading, and dividing the original by it flattens uneven
    lighting toward a uniform white background while leaving text as continuous dark
    values -- no thresholding step at all. CLAHE then adds local contrast punch on top
    of that already-flattened image. `clipLimit` scales with `strength`.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=21)
    normalized = cv2.divide(gray, background, scale=255)

    clip_limit = _lerp(GRAY_CLIP_RANGE, strength)
    if clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)
    return normalized


def _enhance_text_page(warped_bgr: np.ndarray, strength: int = DEFAULT_STRENGTH) -> np.ndarray:
    """Hard black & white: adaptive thresholding lifts text against its local
    background. Adaptive (not global) so uneven lighting across the page doesn't wash
    out one side. This mode inherently produces a jagged, fax-like look -- no midtones,
    every pixel pure black or white -- which is what makes it feel "tough"/harsh; for
    a softer scanned look that keeps anti-aliased edges, see _enhance_gray_page.

    blockSize stays fixed at 61: a smaller window (an earlier attempt used 25) reacted
    to per-pixel JPEG noise and fine text strokes as if they were lighting variation,
    producing heavy black speckle on busy/small-text pages. A light median blur first
    suppresses the same JPEG grain before thresholding sees it. `C` is the part that
    scales with `strength` -- higher C thickens strokes toward pure black/white ("more
    enhanced"), lower C stays closer to the original grayscale contrast.
    """
    c = _lerp(BW_C_RANGE, strength)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=61, C=c,
    )


def _enhance_color_page(warped_bgr: np.ndarray, strength: int = DEFAULT_STRENGTH) -> np.ndarray:
    """Contrast/brightness correction that keeps color -- binarizing a cover or photo
    would destroy the actual content, so this only normalizes lighting and lifts
    contrast instead of thresholding to black & white. No detailEnhance sharpening --
    it read as over-processed/oversaturated next to the original photo instead of a
    natural lighting correction. `clipLimit` is the part that scales with `strength`."""
    clip_limit = _lerp(COLOR_CLIP_RANGE, strength)
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_scan(warped_bgr: np.ndarray, mode: str = "auto", strength: int = DEFAULT_STRENGTH) -> np.ndarray:
    """Turns a flattened (perspective-warped) document photo into a "scanned" look.

    mode: "auto" (default) picks a soft grayscale look for plain text pages and a
    color contrast enhancement for covers/photos, based on median color channel
    spread -- binarizing or de-coloring a book cover throws away the content, so it
    isn't a one-size-fits-all operation. "gray", "bw", and "color" force one path
    regardless of that check: "gray" is the softer, continuous-tone scanned look;
    "bw" is the harder, pure-binary fax-like look for cases that specifically want it.

    strength: 0-100, how aggressively to push contrast/thresholding. 50 matches this
    project's tuned default; 0 is closest to the raw warped image, 100 is the heaviest,
    most "scanner-like" look.
    """
    if mode == "gray":
        return _enhance_gray_page(warped_bgr, strength=strength)
    if mode == "bw":
        return _enhance_text_page(warped_bgr, strength=strength)
    if mode == "color":
        return _enhance_color_page(warped_bgr, strength=strength)
    if mode != "auto":
        raise ValueError(f"Unknown enhance mode: {mode!r}")

    if _is_color_document(warped_bgr):
        return _enhance_color_page(warped_bgr, strength=strength)
    return _enhance_gray_page(warped_bgr, strength=strength)
