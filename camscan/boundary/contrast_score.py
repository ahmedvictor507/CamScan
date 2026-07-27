import cv2
import numpy as np

from camscan.boundary.candidates import quad_candidates


def _boundary_bands(quad, frame_shape, band_width=6):
    """Thin ring of pixels just inside the quad boundary, and another just outside."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)

    kernel = np.ones((band_width, band_width), np.uint8)
    inside_band = mask - cv2.erode(mask, kernel)
    outside_band = cv2.dilate(mask, kernel) - mask
    return inside_band, outside_band


def _contrast(quad, image, band_width=6):
    inside_band, outside_band = _boundary_bands(quad, image.shape, band_width)
    if inside_band.sum() == 0 or outside_band.sum() == 0:
        return -1.0

    inside_mean = np.array(cv2.mean(image, mask=inside_band)[:3])
    outside_mean = np.array(cv2.mean(image, mask=outside_band)[:3])
    return float(np.linalg.norm(inside_mean - outside_mean))


def score_quad(quad, image):
    """Public wrapper around the area-weighted boundary-contrast score, so other
    callers (the fallback chain in pipeline.py) can rank quads from *different*
    detection methods on the same scale, instead of only using this inside
    find_document_contour's own top_n candidate list."""
    frame_area = image.shape[0] * image.shape[1]
    area_ratio = cv2.contourArea(quad.astype(np.float32)) / frame_area
    return _contrast(quad, image) * area_ratio


def find_document_contour(edge_map, image, top_n=15, min_contrast=12):
    """Improvement 2 (Zhukovsky et al., 2020): rather than trusting contour area or
    shape alone, score each candidate quad by how different the pixels just inside its
    border look from the pixels just outside -- a real document boundary is a strong,
    consistent visual seam, whereas an incidental rectangle (an illustration, a sticker,
    a shadow) usually isn't.

    Raw contrast alone still isn't enough: a small high-contrast label (a barcode
    sticker, a strip of colored text) can out-score the document's own, weaker boundary
    against its background, especially under dim/uneven lighting where even the true
    boundary's contrast is weak. Weighting by area helps (sqrt wasn't a strong enough
    penalty), but no single exponent dominates across conditions -- pushing the penalty
    harder (area_ratio**1.5) fixed some low_light/skewed cases while breaking others
    that linear weighting had gotten right. Kept at linear (area_ratio**1) as the best
    single trade-off found; the remaining failures are a real, documented limitation of
    scoring by raw region-mean contrast under uneven lighting, not a tuning gap."""
    quads = quad_candidates(edge_map, top_n=top_n)
    if not quads:
        return None

    frame_area = image.shape[0] * image.shape[1]

    def score(quad):
        area_ratio = cv2.contourArea(quad.astype(np.float32)) / frame_area
        return _contrast(quad, image) * area_ratio

    best_quad = max(quads, key=score)
    if _contrast(best_quad, image) < min_contrast:
        return None
    return best_quad
