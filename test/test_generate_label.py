import numpy as np

from src.generate_dummy_data import generate_data
from src.generate_label import create_image_label


def test_label_direction():
    label = (1, 1, 1)
    data = generate_data(4, 4, 4, *label)
    labels = create_image_label(data, label)
    assert labels[label].tolist() == [0] * 3
    assert np.allclose(labels[0, 0, 0], [1 / 3**0.5] * 3)
    assert np.allclose(labels[2, 2, 2], [-1 / 3**0.5] * 3)
    assert np.allclose(labels[0, 1, 3], [1 / 5**0.5, 0, -2 / 5**0.5])
