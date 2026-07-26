import argparse
from pathlib import Path

import cv2

from camscan.preprocess import resize_for_detection, to_blurred_gray
from camscan.edges import detect_edges
from camscan.boundary.candidates import fallback_frame_contour
from camscan.boundary import baseline, aspect_ratio, contrast_score
from camscan.warp import four_point_transform
from camscan.enhance import enhance_scan

METHODS = {
    "baseline": baseline.find_document_contour,
    "aspect_ratio": aspect_ratio.find_document_contour,
    "contrast_score": contrast_score.find_document_contour,
}
ALL_METHODS = list(METHODS) + ["sam"]  # "sam" is resolved lazily -- see _resolve_method


def _resolve_method(method):
    if method == "sam":
        # deferred: importing torch/mobile_sam costs real time and memory, not worth
        # paying on every run of the (fast, lightweight) classical methods
        from camscan.boundary import sam_boundary
        return sam_boundary.find_document_contour
    return METHODS[method]


def detect_boundary(resized_image, method="baseline"):
    """Returns (quad, used_fallback) in the resized image's coordinate space."""
    gray = to_blurred_gray(resized_image)
    edge_map = detect_edges(gray)
    quad = _resolve_method(method)(edge_map, resized_image)
    if quad is None:
        return fallback_frame_contour(resized_image.shape), True
    return quad, False


def scan(image_path, method="baseline", debug_dir=None):
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(image_path)

    resized, ratio = resize_for_detection(original)
    quad, used_fallback = detect_boundary(resized, method=method)

    if debug_dir is not None:
        overlay = resized.copy()
        cv2.drawContours(overlay, [quad.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(Path(debug_dir) / f"{image_path.stem}_contour.png"), overlay)

    warped = four_point_transform(original, quad * ratio)
    enhanced = enhance_scan(warped)
    return enhanced, used_fallback


def main():
    parser = argparse.ArgumentParser(description="Scan a document photo into a flat, enhanced image.")
    parser.add_argument("image", type=Path, help="Path to the input photo")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <name>_scan.png next to input)")
    parser.add_argument("--method", choices=ALL_METHODS, default="baseline")
    parser.add_argument("--debug-dir", type=Path, default=None, help="If set, save the contour overlay here")
    args = parser.parse_args()

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    result, used_fallback = scan(args.image, method=args.method, debug_dir=args.debug_dir)
    if used_fallback:
        print(f"Warning: no document boundary found, used the full frame as a fallback")

    output_path = args.output or args.image.with_name(f"{args.image.stem}_scan.png")
    cv2.imwrite(str(output_path), result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
