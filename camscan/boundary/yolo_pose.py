import cv2
import numpy as np

CHECKPOINT_PATH = "model/best(3).pt"
_model = None


def _get_model():
    """Lazily loads the YOLO26-pose corner detector -- same lazy-import pattern as
    sam_boundary.py, so importing this module doesn't cost torch/ultralytics init time
    unless this method is actually selected.

    Forced to CPU, same reasoning as sam_boundary.py: this Jetson shares physical RAM
    between CPU and GPU, and under normal desktop load there isn't enough free to grow
    the CUDA allocator (confirmed by a real NvMapMemAllocInternalTagged OOM here)."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(CHECKPOINT_PATH)
        _model.to("cpu")
    return _model


def _order_corners(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left, matching the
    convention the rest of the pipeline (warp.four_point_transform) expects. The model's
    4 keypoints have a fixed semantic order from training but aren't guaranteed to come
    back in TL/TR/BR/BL order for every image, so re-derive it from geometry: TL/BR are
    the smallest/largest coordinate sum, TR/BL are the smallest/largest coordinate
    difference -- same trick used elsewhere for contour-derived quads."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_contour(edge_map, image, conf=0.25, min_kpt_conf=0.5, detection_image=None):
    """Learned-method comparison point: a YOLO26-pose model trained to directly regress
    the 4 document corners (not a mask or an edge-based candidate), at 640x640. Unlike
    every other method in this comparison, it doesn't run candidate generation at all --
    it goes straight from image to 4 keypoints, so it isn't bound by the
    "largest/best-scored candidate quad" blind spot the classical and SAM methods share.

    `edge_map` is accepted but unused, only to match the shared method signature used by
    pipeline.detect_boundary and compare.py. `image` is expected in the shared 500px-wide
    resized coordinate space (like every other method), but this model was trained at
    640x640 and does better given real resolution -- pass the original full-res frame as
    `detection_image` and the returned quad is scaled back into `image`'s coordinate
    space so it stays directly comparable to the other methods' output.
    """
    model = _get_model()
    predict_image = detection_image if detection_image is not None else image
    results = model.predict(predict_image, imgsz=640, conf=conf, verbose=False)[0]

    if results.keypoints is None or len(results.boxes) == 0:
        return None

    # single_cls, one document per image by training design -- take the highest-
    # confidence detection if more than one box ever comes back.
    best_idx = int(results.boxes.conf.argmax())
    kpts_xy = results.keypoints.xy[best_idx].cpu().numpy()
    kpts_conf = results.keypoints.conf[best_idx].cpu().numpy()

    if (kpts_conf < min_kpt_conf).any():
        return None

    quad = _order_corners(kpts_xy.astype(np.float32))

    if detection_image is not None:
        scale_x = image.shape[1] / detection_image.shape[1]
        scale_y = image.shape[0] / detection_image.shape[0]
        quad = quad * np.array([scale_x, scale_y], dtype=np.float32)

    h, w = image.shape[:2]
    if cv2.contourArea(quad) < 0.01 * h * w:
        return None

    return quad
