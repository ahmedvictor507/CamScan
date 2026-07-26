import cv2
import imutils


def resize_for_detection(image, width=500):
    ratio = image.shape[1] / float(width)
    resized = imutils.resize(image, width=width)
    return resized, ratio


def to_blurred_gray(image, kernel_size=(5, 5)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # boosts local contrast so a low-contrast boundary (e.g. a dark object on a dark
    # background) still produces a strong enough gradient for Canny to pick up --
    # a plain global threshold leaves those boundaries invisible
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, kernel_size, 0)
