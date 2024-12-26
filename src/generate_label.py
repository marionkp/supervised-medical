from typing import Tuple

import numpy as np


def create_image_label(image: np.ndarray, label: Tuple[int, int, int]) -> np.ndarray:
    xdim, ydim, zdim = image.shape
    x, y, z = np.meshgrid(np.arange(xdim), np.arange(ydim), np.arange(zdim), indexing="ij")
    dx = label[0] - x
    dy = label[1] - y
    dz = label[2] - z
    diffs = np.stack([dx, dy, dz], axis=-1)
    norm = np.sqrt(np.sum(diffs**2, axis=-1, keepdims=True))
    norm = np.where(norm == 0, 1, norm)
    unit_vectors = diffs / norm
    return unit_vectors
