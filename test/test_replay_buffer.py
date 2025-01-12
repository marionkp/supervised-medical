import numpy as np
import pytest

from src.replay_buffer import ReplayBuffer


def test_replay_buffer_empty():
    rb = ReplayBuffer(100)
    with pytest.raises(Exception):
        rb.sample_roi_and_label(10)


def test_replay_buffer_add_and_sample():
    rb = ReplayBuffer(100)
    image = np.ones((1, 2, 3, 4))
    label = np.ones((1, 3))
    rb.add_to_buffer((image, label))
    sampled_image, sampled_label = rb.sample_roi_and_label(batch_size=1)
    assert (sampled_image == image).all() and (sampled_label == label).all()
