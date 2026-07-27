import numpy as np
import pytest

from camscan.boundary.yolo26_doccorner_pose import _box_iou, _order_corners, _letterbox


class TestOrderCorners:
    def test_already_ordered_stays_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        rect = _order_corners(pts)
        np.testing.assert_allclose(rect, pts)

    def test_scrambled_order_recovers_tl_tr_br_bl(self):
        tl, tr, br, bl = (0, 0), (100, 0), (100, 100), (0, 100)
        # Same 4 points, deliberately given in a different order (BR, TL, BL, TR).
        scrambled = np.array([br, tl, bl, tr], dtype=np.float32)
        rect = _order_corners(scrambled)
        np.testing.assert_allclose(rect[0], tl)
        np.testing.assert_allclose(rect[1], tr)
        np.testing.assert_allclose(rect[2], br)
        np.testing.assert_allclose(rect[3], bl)

    def test_rotated_quad(self):
        # A quad rotated ~30 degrees off-axis -- corner sum/diff heuristic should still
        # recover the correct TL/TR/BR/BL assignment, not just work on axis-aligned boxes.
        pts = np.array([[50, 0], [150, 30], [120, 130], [20, 100]], dtype=np.float32)
        rect = _order_corners(pts)
        # TL is whichever point has the smallest x+y sum.
        sums = pts.sum(axis=1)
        expected_tl = pts[np.argmin(sums)]
        np.testing.assert_allclose(rect[0], expected_tl)


class TestBoxIou:
    def test_identical_boxes_iou_is_one(self):
        box = (0, 0, 100, 100)
        assert _box_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert _box_iou(a, b) == 0.0

    def test_half_overlap(self):
        a = (0, 0, 10, 10)
        b = (5, 0, 15, 10)
        # intersection = 5x10=50, union = 100+100-50=150
        assert _box_iou(a, b) == pytest.approx(50 / 150)

    def test_zero_area_boxes_dont_divide_by_zero(self):
        a = (0, 0, 0, 0)
        b = (0, 0, 0, 0)
        assert _box_iou(a, b) == 0.0


class TestLetterbox:
    def test_square_image_no_padding(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        padded, scale, pad_left, pad_top = _letterbox(image, 800)
        assert padded.shape == (800, 800, 3)
        assert scale == pytest.approx(2.0)
        assert pad_left == 0
        assert pad_top == 0

    def test_wide_image_pads_top_bottom(self):
        # Wider than tall -- letterboxing should scale to fit width and pad vertically,
        # not stretch the aspect ratio.
        image = np.zeros((200, 800, 3), dtype=np.uint8)
        padded, scale, pad_left, pad_top = _letterbox(image, 800)
        assert padded.shape == (800, 800, 3)
        assert scale == pytest.approx(1.0)
        assert pad_left == 0
        assert pad_top == 300  # (800 - 200) / 2

    def test_tall_image_pads_left_right(self):
        image = np.zeros((800, 200, 3), dtype=np.uint8)
        padded, scale, pad_left, pad_top = _letterbox(image, 800)
        assert padded.shape == (800, 800, 3)
        assert scale == pytest.approx(1.0)
        assert pad_left == 300
        assert pad_top == 0
