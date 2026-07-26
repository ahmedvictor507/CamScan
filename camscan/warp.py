import cv2
import numpy as np

# Hough-detected line segments shorter than this fraction of the image width are
# ignored -- short segments are more likely to be individual letters/noise than actual
# text lines or ruled-paper lines, and including them makes the dominant-angle vote
# noisier rather than more accurate.
MIN_LINE_LENGTH_RATIO = 0.25
# A detected skew beyond this is treated as "not really horizontal text" (e.g. a
# vertical ruling, a diagram, a mostly-blank page with only a couple of stray edges)
# rather than force-rotating the page based on an unreliable signal.
MAX_CORRECTABLE_SKEW_DEGREES = 15.0


def _dominant_text_angle(bgr):
    """Estimates how far the page's dominant line direction (text lines, ruled paper,
    table borders) is from perfectly horizontal, in degrees. Returns 0.0 if there
    isn't a clear enough dominant angle to act on."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    min_length = int(bgr.shape[1] * MIN_LINE_LENGTH_RATIO)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80, minLineLength=min_length, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    # OpenCV 4.x returns shape (N, 1, 4); OpenCV 5.x returns (N, 4) directly --
    # reshape unifies both so the unpack below always sees per-line (x1, y1, x2, y2).
    lines = lines.reshape(-1, 4)

    angles = []
    for x1, y1, x2, y2 in lines:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # fold to (-45, 45] -- a text line and the same line rotated 180 degrees are
        # the same skew, and this keeps near-vertical rulings from dragging the
        # histogram toward +/-90 instead of clustering near 0
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) > MAX_CORRECTABLE_SKEW_DEGREES:
        return 0.0
    return median_angle


def _horizontal_line_energy(gray):
    """Scores how strongly a page's dominant line structure (text lines, ruled
    paper, table borders) is horizontal, by summing ink per row and looking at the
    variance across rows -- upright text alternates between text-line rows (high ink)
    and inter-line gaps (near-zero ink), giving a high-variance row profile, while the
    same content rotated 90 degrees smears that structure evenly across every row,
    giving a flat, low-variance profile."""
    edges = cv2.Canny(gray, 50, 150)
    row_ink = edges.sum(axis=1).astype(np.float64)
    return float(np.var(row_ink))


def detect_orientation(bgr):
    """Detects whether the page is sideways (90 or 270 degrees from upright) and
    returns the rotation (0, 90, 180, or 270) needed to make its dominant line
    structure horizontal again -- e.g. if the user held their phone in landscape,
    the warped page comes out with its text running vertically.

    Only resolves the axis (0/180 vs 90/270): comparing horizontal-line energy at the
    page's current orientation against the same page rotated 90 degrees reliably picks
    which axis has the horizontal text-line banding, but a page and its 180-degree
    (upside-down) rotation look identical to this measure -- telling right-side-up
    apart from upside-down needs actual letterform recognition (OCR), which this
    project intentionally leaves as a manual rotate step instead, the same "don't
    force a correction from an unreliable signal" stance _dominant_text_angle takes
    for skew. Returns 0 if the two axes aren't clearly distinguishable (e.g. a mostly
    blank or non-text page), rather than guessing.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    upright_energy = _horizontal_line_energy(gray)
    sideways_energy = _horizontal_line_energy(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE))

    if upright_energy == 0 and sideways_energy == 0:
        return 0
    if sideways_energy > upright_energy * 1.5:
        return 90
    return 0


def rotate_by(bgr, degrees):
    """Rotates by a multiple of 90 degrees (clockwise). No-op for 0."""
    if degrees == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def deskew(bgr):
    """Rotates a warped document image so its dominant text/ruling lines are
    horizontal. The 4-point perspective warp already squares up the page's outer
    boundary into a rectangle, but the detected corners are only an approximation of
    the true edges (more so on a curled/held page than a flat sheet), so the content
    inside can still come out with a residual slant that this corrects separately."""
    angle = _dominant_text_angle(bgr)
    if abs(angle) < 0.5:
        return bgr

    h, w = bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def order_points(pts):
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]        # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]        # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]     # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]     # bottom-left: largest y-x
    return ordered


def four_point_transform(image, pts):
    tl, tr, br, bl = order_points(pts)

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
