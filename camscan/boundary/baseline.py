import numpy as np

from camscan.boundary.candidates import quad_candidates


def find_document_contour(edge_map: np.ndarray, image: np.ndarray | None = None, top_n: int = 5) -> np.ndarray | None:
    """Baseline: just take the largest candidate quad. No content check at all --
    a small high-contrast rectangle (a sticker, a label) beats the actual document
    whenever it happens to be a cleaner quad."""
    quads = quad_candidates(edge_map, top_n)
    return quads[0] if quads else None
