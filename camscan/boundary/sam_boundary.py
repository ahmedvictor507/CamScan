import cv2
import numpy as np

CHECKPOINT_PATH = "models/mobile_sam.pt"
_predictor = None


def _get_predictor():
    """Lazily loads MobileSAM (torch/model init is slow and unnecessary for the
    classical methods) -- a lighter, same-API distillation of Segment Anything, chosen
    over full SAM because this Jetson has under 1GB of RAM free under normal desktop
    load; SAM ViT-B's encoder is heavy enough to risk OOM/thrashing here."""
    global _predictor
    if _predictor is None:
        from mobile_sam import SamPredictor, sam_model_registry

        # Forced to CPU: on this Jetson, CUDA and system RAM share the same physical
        # memory, and under normal desktop load (~1GB free) the GPU allocator fails
        # outright (NVML/CUDACachingAllocator OOM) trying to load the encoder onto the
        # GPU. CPU inference is slower but actually completes.
        device = "cpu"
        model = sam_model_registry["vit_t"](checkpoint=CHECKPOINT_PATH)
        model.to(device=device)
        model.eval()
        _predictor = SamPredictor(model)
    return _predictor


def _largest_contour(mask):
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _mask_to_quad(mask, contour):
    # a segmentation mask's outline isn't guaranteed to already be a clean polygon
    # (rounded corners, a slightly curled page edge) the way a Canny contour's
    # approxPolyDP candidates are expected to be -- minAreaRect fits the tightest
    # rotated rectangle around the mask instead of requiring the outline itself to
    # already look like a quad
    rect = cv2.minAreaRect(cv2.convexHull(contour))
    return rect, cv2.boxPoints(rect)


def _rect_fill_ratio(mask, rect):
    """actual foreground pixel count vs. the area of the mask's own minimum bounding
    rotated rectangle. Close to 1.0 for a solid rectangular blob. With a box prompt
    that covers most of the image, SAM sometimes segments the background frame *around*
    the document instead -- a concave, "C"-shaped region (three sides of a border) that
    still traces a single clean, simply-connected contour with no actual hole, so a
    naive solidity check (mask area / its own contour area) scores it near 1.0 too and
    can't tell the two apart. Its own bounding rectangle can: the C-shape only fills
    about half of the smallest rectangle that contains it, while a real document
    fills nearly all of it."""
    rect_area = rect[1][0] * rect[1][1]
    return mask.sum() / rect_area if rect_area > 0 else 0.0


def find_document_contour(edge_map, image, box_margin=0.05, min_area_ratio=0.05, min_rect_fill=0.85):
    """Stretch goal: MobileSAM with a box prompt covering most of the frame, as a
    learned-method comparison point against the classical candidate-and-rescore
    methods. Prompted with a large centered box so SAM segments the single dominant
    object in frame rather than the background -- it doesn't depend on Canny/contour
    candidate generation at all, so it isn't bound by the "largest candidate vs a stack
    of documents" blind spot the other three share.

    Doesn't just take SAM's top-scoring mask: with a box that covers most of the image,
    SAM sometimes segments the background frame around the document and scores it
    higher than the document itself. Filtering by rect_fill first, then ranking by
    SAM's own score among what's left, avoids that.

    Tried augmenting this with a second, tighter box prompt seeded from the classical
    candidate generator's rough localization, pooling both boxes' candidates and
    picking the overall highest score. Reverted: SAM's confidence score isn't
    comparable across different box prompts -- a looser, less accurate candidate from
    the second box sometimes scored higher than the correct one from this box, so
    pooling introduced new confident-wrong-answers (previously honest fallbacks on
    cluttered_06 and low_light_06 became wrong-but-claimed-successful) for one modest,
    still-imperfect partial fix (skewed_02). Net effect was negative on a full audit;
    single box prompt is the better-understood, honestly-audited version."""
    predictor = _get_predictor()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    h, w = image.shape[:2]
    mx, my = int(w * box_margin), int(h * box_margin)
    box = np.array([mx, my, w - mx, h - my])

    masks, scores, _ = predictor.predict(box=box, multimask_output=True)

    frame_area = h * w
    candidates = []
    for mask, score in zip(masks, scores):
        contour = _largest_contour(mask)
        if contour is None or mask.sum() < min_area_ratio * frame_area:
            continue
        rect, quad = _mask_to_quad(mask, contour)
        if _rect_fill_ratio(mask, rect) < min_rect_fill:
            continue
        candidates.append((score, quad))

    if not candidates:
        return None

    _, best_quad = max(candidates, key=lambda c: c[0])
    return best_quad
