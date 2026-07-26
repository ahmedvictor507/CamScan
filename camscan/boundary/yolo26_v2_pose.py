import cv2
import numpy as np

CHECKPOINT_PATH = "model/attempt 3_yolo26/weights/best.pt"
_model = None


def _get_model():
    """Lazily loads this YOLO26n-pose corner detector -- same lazy-import pattern as
    the other learned methods, so importing this module doesn't cost torch/ultralytics
    init time unless this method is actually selected.

    Forced to CPU, same reasoning as the other learned methods: this Jetson shares
    physical RAM between CPU and GPU, and under normal desktop load there isn't enough
    free to grow the CUDA allocator (confirmed by a real NvMapMemAllocInternalTagged OOM
    on this box)."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(CHECKPOINT_PATH)
        _model.to("cpu")
    return _model


def _order_corners(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left, matching the
    convention warp.four_point_transform expects. Re-derived from geometry (TL/BR =
    smallest/largest coordinate sum, TR/BL = smallest/largest coordinate difference)
    since the model's keypoint order isn't guaranteed to be TL/TR/BR/BL for every image."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_contour(edge_map, image, conf=0.25, detection_image=None):
    """A third training attempt: another YOLO26n-pose run (same dataset/task as
    yolo_pose.py's attempt-1 run, which suffered a 2-point keypoint collapse), this time
    50 epochs with AdamW and rle=1.5. Spot-checked before wiring in: keypoints come back
    genuinely distinct (4 real corners, not collapsed), so this is evaluated the same
    direct way as yolo_v8_pose.py -- no classical refinement layer.

    Does not gate on a per-keypoint confidence floor (same reasoning as
    yolo_v8_pose.py): per-point confidence on these checkpoints is noisy even on
    otherwise-correct corners, so only the overall box confidence (`conf`) filters
    detections.

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

    best_idx = int(results.boxes.conf.argmax())
    kpts_xy = results.keypoints.xy[best_idx].cpu().numpy()

    quad = _order_corners(kpts_xy.astype(np.float32))

    if detection_image is not None:
        scale_x = image.shape[1] / detection_image.shape[1]
        scale_y = image.shape[0] / detection_image.shape[0]
        quad = quad * np.array([scale_x, scale_y], dtype=np.float32)

    h, w = image.shape[:2]
    if cv2.contourArea(quad) < 0.01 * h * w:
        return None

    return quad
