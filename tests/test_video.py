"""Debug-video helpers: depth colouring, top-down map pixel mapping and frame composition."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kznav.report.video import TopDownMap, compose, depth_to_rgb  # noqa: E402


def test_depth_to_rgb_marks_invalid_black():
    d = np.array([[0.0, 1.0], [5.0, 10.0]], dtype=np.float32)
    img = depth_to_rgb(d, max_m=5.0)
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    assert (img[0, 0] == 0).all()
    assert not (img[0, 1] == img[1, 0]).all()     # near and far get different colours


def test_topdown_pixel_mapping():
    m = TopDownMap(np.ones((100, 200), dtype=bool), lower_x=-1.0, lower_z=-2.0, mpp=0.05)
    assert m.to_px(-1.0, 2.0) == (0, 0)            # map y = -z
    assert m.to_px(0.0, 0.0) == (20, 40)
    img = m.render(path=[(0, 0), (1, -1)], trajectory=[(0, 0), (0.5, -0.5)], pose=(0.5, -0.5, 0.0), goal=(1, -1))
    assert img.shape == (100, 200, 3)


def test_compose_even_dims_and_band():
    a = np.zeros((481, 641, 3), dtype=np.uint8)
    b = np.zeros((101, 51, 3), dtype=np.uint8)
    f = compose([a, b], height=361, text_lines=["one", "two"])
    assert f.shape[0] % 2 == 0 and f.shape[1] % 2 == 0
    assert f.shape[0] >= 361
