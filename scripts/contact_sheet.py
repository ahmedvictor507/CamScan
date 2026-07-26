import sys
from pathlib import Path

import cv2
import numpy as np

DEBUG_DIR = Path("data/results/debug")
OUT_DIR = Path("data/results/contact_sheets")
THUMB_SIZE = (220, 165)
COLS = 4


def make_sheet(method, condition):
    images_dir = DEBUG_DIR / method / condition
    paths = sorted(images_dir.glob("*_contour.png"))
    if not paths:
        return

    thumbs = []
    for p in paths:
        img = cv2.resize(cv2.imread(str(p)), THUMB_SIZE)
        label = p.stem.replace("_contour", "")
        cv2.rectangle(img, (0, 0), (THUMB_SIZE[0], 18), (0, 0, 0), -1)
        cv2.putText(img, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        thumbs.append(img)

    rows = -(-len(thumbs) // COLS)
    sheet = np.zeros((rows * THUMB_SIZE[1], COLS * THUMB_SIZE[0], 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, COLS)
        sheet[r*THUMB_SIZE[1]:(r+1)*THUMB_SIZE[1], c*THUMB_SIZE[0]:(c+1)*THUMB_SIZE[0]] = thumb

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{method}_{condition}.png"
    cv2.imwrite(str(out_path), sheet)
    print(out_path)


if __name__ == "__main__":
    methods = ["baseline", "aspect_ratio", "contrast_score", "sam", "yolo_pose", "yolo_hybrid", "yolo_v8_pose"]
    conditions = ["clean", "cluttered", "low_light", "skewed"]
    for method in methods:
        for condition in conditions:
            make_sheet(method, condition)
