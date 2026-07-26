import cv2
import numpy as np


def detect_edges(gray, sigma=0.33):
    # thresholds relative to the image's own median intensity, instead of one fixed
    # global pair -- a threshold tuned for a high-contrast photo silently misses the
    # boundary in a low-contrast one (dark document on a dark background)
    median = np.median(gray)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(gray, low, high)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
