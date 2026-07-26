import cv2
import numpy as np


def detect_edges(gray, low=75, high=200):
    edges = cv2.Canny(gray, low, high)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
