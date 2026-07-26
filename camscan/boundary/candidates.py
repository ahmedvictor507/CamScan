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
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) >= min_area_ratio * frame_area:
                quads.append(approx.reshape(4, 2))
    return quads


def fallback_frame_contour(image_shape):
    h, w = image_shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
