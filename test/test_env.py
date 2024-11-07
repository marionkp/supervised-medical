import numpy as np

from src.env import dist_3d_points, get_roi_from_image, get_roi_dims_from_len_and_stride


def test_roi_is_full_image():
    """
    x x x x x
    x x P x x
    x x x x x
    (first 2 dimensions showed)
    """
    pos = (1, 2, 3)
    image = np.random.randn(3, 5, 7)
    roi_len = (2, 3, 4)
    roi = get_roi_from_image(pos, image, roi_len, 1)
    assert (roi == image).all()


def test_roi_with_padding():
    """
    0 0 0 0 0
    0 x P 0 0
    0 0 0 0 0
    """
    pos = (0, 1, 2)
    image = np.random.randn(1, 2, 3)
    roi_len = (2, 3, 4)
    roi = get_roi_from_image(pos, image, roi_len, 1)
    assert roi.shape == (3, 5, 7)
    assert (roi[1:2, 1:3, 1:4] == image).all()
    assert roi[0, 0, 0] == roi[2, 3, 4] == 0


def test_roi_with_padding_and_stride():
    pos = (0, 1, 2)
    image = np.random.randn(1, 2, 3)
    roi_len = (2, 3, 4)
    stride = 3
    roi = get_roi_from_image(pos, image, roi_len, stride)
    assert roi.shape == get_roi_dims_from_len_and_stride(roi_len, stride)
    assert (roi[1:2:stride, 1:3:stride, 1:4:stride] == image).all()
    assert roi[0, 0, 0] == roi[-1, -1, -1] == 0


def test_3d_point_dist():
    assert dist_3d_points((0, 0, 0), (0, -1, 0)) == 1
    assert dist_3d_points((1, 4, 5), (1, 1, 1)) == 5


# TODO: add test to check the right samples are given to the replay buffer during the episode

# TODO: check if agent within image of size 2x2x2 gets perfect accuracy
