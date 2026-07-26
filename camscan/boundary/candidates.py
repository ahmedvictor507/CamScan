import cv2
import imutils
import numpy as np


def quad_candidates(edge_map, top_n=10, min_area_ratio=0.02):
    """Largest closed contours in the edge map that reduce to a convex 4-point polygon,
    ordered by area (largest first). Candidates smaller than min_area_ratio of the frame
    are dropped -- approxPolyDP can collapse a noisy sliver contour into a technically
    convex but near-zero-area quad, which isContourConvex alone won't catch."""
    frame_area = edge_map.shape[0] * edge_map.shape[1]
    contours = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    quads = []
    for contour in contours:
        # the true boundary is often traced as a noisy, slightly non-convex polygon
        # (broken edge segments, texture inside the document) that approxPolyDP can't
        # collapse to 4 points directly -- taking the convex hull first strips that
        # noise out, since a document's outline is convex by construction
        hull = cv2.convexHull(contour)
        approx = _approx_to_quad(hull)
        if approx is not None and cv2.contourArea(approx) >= min_area_ratio * frame_area:
            quads.append(approx.reshape(4, 2))
    return quads


def _approx_to_quad(hull, start=0.02, stop=0.10, step=0.01):
    """approxPolyDP at a fixed epsilon often leaves a document's outline at 5-6 points
    (one corner slightly rounded or doubled) even after the convex hull cleanup --
    stepping the epsilon up until it collapses to exactly 4 recovers those near-quads
    instead of discarding them."""
    perimeter = cv2.arcLength(hull, True)
    epsilon = start
    while epsilon <= stop:
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4:
            return approx if cv2.isContourConvex(approx) else None
        epsilon += step
    return None


def fallback_frame_contour(image_shape):
    h, w = image_shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
