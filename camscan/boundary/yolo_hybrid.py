import cv2
import numpy as np

from camscan.boundary.candidates import quad_candidates
from camscan.boundary import yolo_pose
from camscan.edges import detect_edges
from camscan.preprocess import to_blurred_gray


def _yolo_box(image, conf=0.25):
    """Runs the YOLO26-pose detector and returns just its bounding box in `image`'s own
    coordinate space, or None. The model's 4 keypoints collapse to 2 distinct points
    (see yolo_pose.py) -- a real dataset/export issue, not usable for a direct 4-point
    warp -- but the box itself is a normal, reliable detection head output and isn't
    affected by that problem."""
    model = yolo_pose._get_model()
    results = model.predict(image, imgsz=640, conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    best_idx = int(results.boxes.conf.argmax())
    return results.boxes.xyxy[best_idx].cpu().numpy()


def find_document_contour(edge_map, image, detection_image=None, box_margin=0.08, min_area_ratio=0.35):
    """Hybrid method: use YOLO26-pose's bounding box (its one reliable output, see
    yolo_pose.py) to localize and crop to the document first, then run the same
    classical Canny + convex-hull + approxPolyDP candidate search the other methods use
    -- but inside the crop, where the document is the dominant shape in frame instead of
    competing with background clutter or a messy tabletop. This is the "learned coarse
    localization + classical fine boundary" combination the user asked for once the
    model's own keypoints turned out to be unusable for a direct 4-point warp.

    `edge_map` is accepted but unused (a fresh edge map is computed on the crop, since
    the shared one was built for the whole frame at the wrong scale). `image` is in the
    shared 500px-wide resized coordinate space; `detection_image` (full-res original) is
    what YOLO actually runs on, same convention as yolo_pose.find_document_contour.
    """
    predict_image = detection_image if detection_image is not None else image
    box = _yolo_box(predict_image)
    if box is None:
        return None

    scale_x = image.shape[1] / predict_image.shape[1]
    scale_y = image.shape[0] / predict_image.shape[0]
    x1, y1, x2, y2 = box * np.array([scale_x, scale_y, scale_x, scale_y])

    h, w = image.shape[:2]
    mx, my = (x2 - x1) * box_margin, (y2 - y1) * box_margin
    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w, int(x2 + mx))
    cy2 = min(h, int(y2 + my))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = image[cy1:cy2, cx1:cx2]
    crop_gray = to_blurred_gray(crop)
    crop_edges = detect_edges(crop_gray)

    quads = quad_candidates(crop_edges, min_area_ratio=min_area_ratio)
    if not quads:
        return None

    quad = max(quads, key=cv2.contourArea)
    return quad + np.array([cx1, cy1], dtype=np.float32)
