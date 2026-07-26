import cv2
import numpy as np


def order_points(pts):
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]        # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]        # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]     # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]     # bottom-left: largest y-x
    return ordered


def four_point_transform(image, pts):
    tl, tr, br, bl = order_points(pts)

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
