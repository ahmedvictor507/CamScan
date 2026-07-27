import cv2
import numpy as np

CHECKPOINT_PATH = "model/attempt 5_yolo26s/weights/best.onnx"
# Trained at imgsz=800 (see model/attempt 5_yolo26s/args.yaml) -- unlike attempt 4
# (640), inference must be run at the same size the model was trained/exported at.
PREDICT_IMGSZ = 800
_session = None

# Two detected boxes with IoU above this are treated as the same physical document
# (near-duplicate detections) rather than two separate documents in a batch photo --
# empirically, the model's own baked-in NMS (opset export default) doesn't merge every
# near-duplicate this model produces; see find_document_contours' docstring for a real
# example where two boxes over the same page survived default NMS with only moderate
# overlap.
BATCH_DEDUP_IOU_THRESHOLD = 0.3


def _get_session():
    """Lazily loads this YOLO26s-pose corner detector via raw ONNX Runtime -- same
    lazy-import pattern as the other learned methods, so importing this module doesn't
    cost onnxruntime init time unless this method is actually selected.

    Runs inference through onnxruntime.InferenceSession directly instead of the
    ultralytics.YOLO wrapper used during training/comparison -- ultralytics pulls in
    torch+torchvision as hard dependencies (~4GB) purely to do letterbox/NMS
    bookkeeping this file reimplements itself in ~30 lines (see _preprocess/_run_model
    below), which matters for keeping the deployed container image small. Preprocessing
    (letterbox resize + pad) and postprocessing (box/keypoint coordinate unletterboxing)
    here were verified to numerically match ultralytics' own output on this checkpoint
    to 3+ decimal places before this rewrite -- box and keypoint coordinates are
    reproduced exactly, not approximately."""
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(CHECKPOINT_PATH, providers=["CPUExecutionProvider"])
    return _session


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


def _letterbox(image, size):
    """Resizes `image` to fit within size x size preserving aspect ratio, then pads
    with 114-gray to a square -- matches ultralytics' own default preprocessing
    exactly (verified numerically against ultralytics.YOLO's output on this
    checkpoint), which is what this model was trained/exported expecting as input.
    Returns (padded_image, scale, pad_left, pad_top) so detections can be mapped back
    into the original image's coordinate space afterward.
    """
    h0, w0 = image.shape[:2]
    scale = min(size / h0, size / w0)
    new_w, new_h = round(w0 * scale), round(h0 * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = size - new_w, size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top


def _run_model(image, detection_image, conf):
    session = _get_session()
    predict_image = detection_image if detection_image is not None else image

    padded, scale, pad_left, pad_top = _letterbox(predict_image, PREDICT_IMGSZ)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = np.transpose(rgb, (2, 0, 1))[None]

    output = session.run(None, {"images": input_tensor})[0][0]  # (300, 18): NMS already applied by the exported graph

    confs = output[:, 4]
    keep = confs >= conf
    output = output[keep]
    confs = confs[keep]
    if len(output) == 0:
        return []

    order = np.argsort(-confs)
    output = output[order]

    boxes_letterboxed = output[:, :4]
    kpts_letterboxed = output[:, 6:].reshape(-1, 4, 3)[:, :, :2]  # drop per-keypoint visibility, unused (see docstring below)

    boxes_xyxy = boxes_letterboxed.copy()
    boxes_xyxy[:, [0, 2]] = (boxes_letterboxed[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_letterboxed[:, [1, 3]] - pad_top) / scale

    kpts = kpts_letterboxed.copy()
    kpts[:, :, 0] = (kpts_letterboxed[:, :, 0] - pad_left) / scale
    kpts[:, :, 1] = (kpts_letterboxed[:, :, 1] - pad_top) / scale

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
    """Attempt 5: YOLO26s-pose (larger backbone than attempt 4's YOLO26n), trained on
    the same Hugging Face DocCornerDataset lineage as attempt 4. Head-to-head against
    attempt 4 on this project's 38-image test set: near-identical raw detection rate
    (27/38 vs attempt 4's 28/38) but visibly tighter/cleaner quads on shared hits
    (skewed and cluttered images especially) -- see docs/progress_log.md for the full
    comparison. Chosen over attempt 4 for the quad-quality edge plus the ONNX export
    path (see _get_session's docstring), not because it wins on recall.

    Same interface and full-res-input convention as the other learned methods:
    predicts at 800x800 (PREDICT_IMGSZ, matching this model's training imgsz -- see
    model/attempt 5_yolo26s/args.yaml) on the original image, scales the resulting quad
    back into the shared 500px detection space. No per-keypoint confidence gate (same
    reasoning as the other direct-keypoint methods) -- only overall box confidence
    (`conf`) filters detections.

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
    box) that the exported graph's own NMS did not merge -- so this function adds its
    own stricter IoU-based de-duplication (BATCH_DEDUP_IOU_THRESHOLD=0.3) before
    treating multiple boxes as multiple documents. Whether the model can actually
    *separate* two distinct side-by-side documents (rather than just duplicate-detect
    one) is untested; treat results as a starting point for manual review, not a
    trusted final answer, until tried on a real multi-doc batch photo.
    """
    return _run_model(image, detection_image, conf)
