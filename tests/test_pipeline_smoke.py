from pathlib import Path

import numpy as np
import pytest

from camscan.pipeline import scan

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.mark.parametrize("image_name", ["clean/clean_09.jpeg"])
def test_scan_end_to_end_produces_a_sane_image(image_name):
    """Not a detection-accuracy check (see compare.py/docs/progress_log.md for that) --
    just confirms the full detect -> warp -> enhance pipeline runs to completion on a
    real photo without crashing and returns a plausible image, catching "the whole
    thing is broken" regressions cheaply."""
    image_path = DATA_DIR / image_name
    if not image_path.exists():
        pytest.skip(f"test image not present: {image_path}")

    result, used_fallback = scan(image_path)

    assert isinstance(result, np.ndarray)
    assert result.ndim in (2, 3)  # grayscale/bw or color
    assert result.shape[0] > 0 and result.shape[1] > 0
    assert isinstance(used_fallback, bool)
