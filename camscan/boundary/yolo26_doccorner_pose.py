import cv2
import numpy as np

CHECKPOINT_PATH = "model/attempt 4_yolo26/weights/best.pt"
_model = None

# Two detected boxes with IoU above this are treated as the same physical document
# (near-duplicate detections) rather than two separate documents in a batch photo --
# empirically, Ultralytics' own NMS (iou=0.7 default) doesn't merge every near-duplicate
# this model produces; see find_document_contours' docstring for a real example where
# two boxes over the same page survived default NMS with only moderate overlap.
BATCH_DEDUP_IOU_THRESHOLD = 0.3


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


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _run_model(image, detection_image, conf):
    model = _get_model()
    predict_image = detection_image if detection_image is not None else image
    results = model.predict(predict_image, imgsz=640, conf=conf, verbose=False)[0]

    if results.keypoints is None or len(results.boxes) == 0:
        return []

    order = results.boxes.conf.argsort(descending=True)
    boxes_xyxy = results.boxes.xyxy[order].cpu().numpy()
    kpts = results.keypoints.xy[order].cpu().numpy()

    kept_boxes = []
    kept_quads = []
    for box, kpt in zip(boxes_xyxy, kpts):
        if any(_box_iou(box, kept) > BATCH_DEDUP_IOU_THRESHOLD for kept in kept_boxes):
            continue
        kept_boxes.append(box)
        quad = _order_corners(kpt.astype(np.float32))

        if detection_image is not None:
            scale_x = image.shape[1] / detection_image.shape[1]
            scale_y = image.shape[0] / detection_image.shape[0]
            quad = quad * np.array([scale_x, scale_y], dtype=np.float32)

        h, w = image.shape[:2]
        if cv2.contourArea(quad) < 0.01 * h * w:
            continue
        kept_quads.append(quad)

    return kept_quads


def find_document_contour(edge_map, image, conf=0.25, detection_image=None):
    """Attempt 4: YOLO26n-pose retrained from scratch on 3000 images from the
    Hugging Face DocCornerDataset (general document-corner images: varied
    backgrounds/lighting/document types) instead of this project's original 500-image,
    mostly-white-paper Roboflow set -- the dataset swap discussed after attempt 3's
    `skewed`/backlit failures looked like a coverage gap rather than an epoch-count
    problem. Reported training metrics are strong (pose mAP50-95 ~0.97), but that's
    measured against DocCornerDataset's own held-out images, not this project's actual
    test photos -- a real domain-shift risk this evaluation exists to catch.

    Same interface and full-res-input convention as yolo_v8_pose.py /
    yolo26_v2_pose.py: predicts at 640x640 on the original image, scales the resulting
    quad back into the shared 500px detection space. No per-keypoint confidence gate
    (same reasoning as the other direct-keypoint methods) -- only overall box
    confidence (`conf`) filters detections.

    `edge_map` is accepted but unused, only to match the shared method signature used
    by pipeline.detect_boundary and compare.py.
    """
    quads = _run_model(image, detection_image, conf)
    return quads[0] if quads else None


def find_document_contours(image, conf=0.25, detection_image=None):
    """Batch variant: returns every distinct document quad detected in one photo,
    instead of just the single best one find_document_contour keeps.

    EXPERIMENTAL / unverified: this project has no real multi-document test photos to
    validate against yet. What's confirmed so far, on a single available cluttered
    photo containing one real document (cluttered_06): the model returned 2 boxes over
    the *same* physical page (xyxy corners within ~50px of each other on a ~1100px-wide
    box) that Ultralytics' own default NMS (iou=0.7) did not merge -- so this function
    adds its own stricter IoU-based de-duplication (BATCH_DEDUP_IOU_THRESHOLD=0.3)
    before treating multiple boxes as multiple documents. Whether the model can
    actually *separate* two distinct side-by-side documents (rather than just
    duplicate-detect one) is untested; treat results as a starting point for manual
    review, not a trusted final answer, until tried on a real multi-doc batch photo.
    """
    return _run_model(image, detection_image, conf)
