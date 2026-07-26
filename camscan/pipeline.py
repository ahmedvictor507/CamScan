import argparse
from pathlib import Path

import cv2

from camscan.preprocess import resize_for_detection, to_blurred_gray
from camscan.edges import detect_edges
from camscan.boundary.baseline import find_document_contour
from camscan.warp import four_point_transform
from camscan.enhance import enhance_scan


def scan(image_path, debug_dir=None):
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(image_path)

    resized, ratio = resize_for_detection(original)
    gray = to_blurred_gray(resized)
    edge_map = detect_edges(gray)
    contour = find_document_contour(edge_map, resized.shape)

    if debug_dir is not None:
        overlay = resized.copy()
        cv2.drawContours(overlay, [contour.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(str(Path(debug_dir) / f"{image_path.stem}_edges.png"), edge_map)
        cv2.imwrite(str(Path(debug_dir) / f"{image_path.stem}_contour.png"), overlay)

    warped = four_point_transform(original, contour * ratio)
    enhanced = enhance_scan(warped)
    return enhanced


def main():
    parser = argparse.ArgumentParser(description="Scan a document photo into a flat, enhanced image.")
    parser.add_argument("image", type=Path, help="Path to the input photo")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <name>_scan.png next to input)")
    parser.add_argument("--debug-dir", type=Path, default=None, help="If set, save the edge map and contour overlay here")
    args = parser.parse_args()

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    result = scan(args.image, debug_dir=args.debug_dir)

    output_path = args.output or args.image.with_name(f"{args.image.stem}_scan.png")
    cv2.imwrite(str(output_path), result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
