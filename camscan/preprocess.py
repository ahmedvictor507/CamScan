import cv2
import imutils


def resize_for_detection(image, width=500):
    ratio = image.shape[1] / float(width)
    resized = imutils.resize(image, width=width)
    return resized, ratio


def to_blurred_gray(image, kernel_size=(5, 5)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, kernel_size, 0)
