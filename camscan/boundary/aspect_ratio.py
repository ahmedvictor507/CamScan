import numpy as np

from camscan.boundary.candidates import quad_candidates
from camscan.warp import order_points

# long-side/short-side ratios for common document formats (portrait A4, portrait Letter)
TARGET_RATIOS = [297 / 210, 11 / 8.5]


def _ratio_deviation(quad: np.ndarray) -> float:
    tl, tr, br, bl = order_points(quad)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    short_side = min(width, height)
    if short_side < 1:
        return float("inf")
    ratio = max(width, height) / short_side
    return min(abs(ratio - target) for target in TARGET_RATIOS)


def find_document_contour(edge_map: np.ndarray, image: np.ndarray | None = None, top_n: int = 10, tolerance: float = 0.35) -> np.ndarray | None:
    """Improvement 1: among the largest candidate quads, prefer the one whose
    aspect ratio is closest to a known document format (A4/Letter), instead of
    blindly taking the single largest quad. Rejects the whole frame if nothing
    is even close -- better to signal failure than warp a wrong shape."""
    quads = quad_candidates(edge_map, top_n)
    if not quads:
        return None

    best = min(quads, key=_ratio_deviation)
    if _ratio_deviation(best) > tolerance:
        return None
    return best
