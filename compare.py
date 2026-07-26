import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from camscan.pipeline import ALL_METHODS, detect_boundary
from camscan.preprocess import resize_for_detection

DATA_DIR = Path("data/raw")
DEBUG_DIR = Path("data/results/debug")
MIN_AREA_RATIO = 0.10  # below this, the quad is too small/degenerate to be the document
MAX_AREA_RATIO = 0.98  # above this, it's basically just the whole frame


def quad_area_ratio(quad, frame_shape):
    frame_area = frame_shape[0] * frame_shape[1]
    return cv2.contourArea(quad.astype(np.float32)) / frame_area


def run_comparison():
    results = []
    for condition_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        condition = condition_dir.name
        for image_path in sorted(condition_dir.glob("*.jpeg")):
            original = cv2.imread(str(image_path))
            resized, _ = resize_for_detection(original)

            for method in ALL_METHODS:
                quad, used_fallback = detect_boundary(resized, method=method)
                area_ratio = quad_area_ratio(quad, resized.shape)
                success = (not used_fallback) and (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO)

                results.append({
                    "condition": condition, "image": image_path.name, "method": method,
                    "success": success, "used_fallback": used_fallback,
                    "area_ratio": round(area_ratio, 3),
                })

                out_dir = DEBUG_DIR / method / condition
                out_dir.mkdir(parents=True, exist_ok=True)
                overlay = resized.copy()
                color = (0, 255, 0) if success else (0, 0, 255)
                cv2.drawContours(overlay, [quad.astype(int)], -1, color, 2)
                cv2.imwrite(str(out_dir / f"{image_path.stem}_contour.png"), overlay)
    return results


def summarize(results):
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # method -> condition -> [success, total]
    for r in results:
        bucket = counts[r["method"]][r["condition"]]
        bucket[0] += r["success"]
        bucket[1] += 1
    return counts


def print_summary(counts):
    conditions = sorted({c for m in counts.values() for c in m})
    header = f"{'method':<14}" + "".join(f"{c:>14}" for c in conditions) + f"{'overall':>14}"
    print(header)
    for method, per_condition in counts.items():
        row = f"{method:<14}"
        total_success = total_n = 0
        for condition in conditions:
            success, n = per_condition.get(condition, [0, 0])
            total_success += success
            total_n += n
            row += f"{f'{success}/{n}':>14}"
        row += f"{f'{total_success}/{total_n}':>14}"
        print(row)


def main():
    results = run_comparison()
    counts = summarize(results)
    print_summary(counts)

    Path("data/results").mkdir(exist_ok=True)
    with open("data/results/comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nPer-image results saved to data/results/comparison.json")
    print("Debug overlays saved to data/results/debug/<method>/<condition>/")


if __name__ == "__main__":
    main()
