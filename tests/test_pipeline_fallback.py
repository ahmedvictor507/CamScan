import numpy as np
import pytest

from camscan import pipeline


def _quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


@pytest.fixture
def blank_image():
    # to_blurred_gray/detect_edges just need *some* valid image to run on -- their
    # output isn't asserted on here, only detect_boundary's method-selection logic is.
    return np.zeros((375, 500, 3), dtype=np.uint8)


def test_primary_method_hit_skips_fallback_entirely(monkeypatch, blank_image):
    primary_quad = _quad(10, 10, 90, 90)
    monkeypatch.setitem(pipeline.METHODS, "baseline", lambda edge_map, image: primary_quad)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fallback method should not run when the primary method succeeds")

    monkeypatch.setitem(pipeline.METHODS, "aspect_ratio", fail_if_called)
    monkeypatch.setitem(pipeline.METHODS, "contrast_score", fail_if_called)

    quad, used_fallback = pipeline.detect_boundary(blank_image, method="baseline")
    np.testing.assert_array_equal(quad, primary_quad)
    assert used_fallback is False


def test_fallback_picks_best_scoring_quad_not_first_runner(monkeypatch, blank_image):
    """Regression test for the exact bug the score-and-pick rework fixed: a
    first-non-None-wins chain would return `worse_quad` here (baseline runs first in
    FALLBACK_CHAIN), even though `better_quad` scores higher. See FALLBACK_CHAIN's
    docstring in pipeline.py and the clean_02 case in docs/progress_log.md.
    """
    monkeypatch.setitem(pipeline.METHODS, "baseline", lambda edge_map, image: None)

    worse_quad = _quad(10, 10, 90, 90)
    better_quad = _quad(5, 5, 95, 95)

    monkeypatch.setitem(pipeline.METHODS, "aspect_ratio", lambda edge_map, image: worse_quad)
    monkeypatch.setitem(pipeline.METHODS, "contrast_score", lambda edge_map, image: better_quad)

    # score_quad scores by (inside/outside contrast) * area_ratio -- on a blank image
    # contrast is ~0 for both, so patch it directly to make the "better" quad the
    # unambiguous winner regardless of image content.
    def fake_score_quad(quad, image):
        return 100.0 if np.array_equal(quad, better_quad) else 1.0

    monkeypatch.setattr(pipeline, "score_quad", fake_score_quad)

    quad, used_fallback = pipeline.detect_boundary(blank_image, method="baseline")
    np.testing.assert_array_equal(quad, better_quad)
    assert used_fallback is True


def test_fallback_skips_the_primary_method_if_also_in_chain(monkeypatch, blank_image):
    """If the primary method is itself one of FALLBACK_CHAIN's methods, it shouldn't
    be re-run a second time during fallback (it already failed once)."""
    calls = []

    def counting_baseline(edge_map, image):
        calls.append("baseline")
        return None

    monkeypatch.setitem(pipeline.METHODS, "baseline", counting_baseline)
    monkeypatch.setitem(pipeline.METHODS, "aspect_ratio", lambda edge_map, image: None)
    monkeypatch.setitem(pipeline.METHODS, "contrast_score", lambda edge_map, image: None)

    pipeline.detect_boundary(blank_image, method="baseline")
    assert calls.count("baseline") == 1


def test_full_frame_fallback_when_every_method_fails(monkeypatch, blank_image):
    monkeypatch.setitem(pipeline.METHODS, "baseline", lambda edge_map, image: None)
    monkeypatch.setitem(pipeline.METHODS, "aspect_ratio", lambda edge_map, image: None)
    monkeypatch.setitem(pipeline.METHODS, "contrast_score", lambda edge_map, image: None)

    quad, used_fallback = pipeline.detect_boundary(blank_image, method="baseline")
    assert used_fallback is True
    h, w = blank_image.shape[:2]
    # full-frame fallback should cover (approximately) the whole resized image
    assert quad[:, 0].max() == pytest.approx(w, rel=0.05)
    assert quad[:, 1].max() == pytest.approx(h, rel=0.05)
