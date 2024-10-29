from typing import Tuple

import numpy as np


def create_empty_image_label(image: np.ndarray, label: Tuple[int, int, int]) -> np.ndarray:
    assert len(image.shape) == 3
    labels = np.zeros_like(image)
    labels = np.repeat(labels[:, :, :, np.newaxis], repeats=3, axis=3)
    assert len(labels.shape) == 4
    assert labels.shape == (*image.shape, 3)
    return labels


# TODO: should the direction be a unit vector directly towards the landmark instead?
def get_direction_to_label_discret(point: Tuple[int, int, int], label: Tuple[int, int, int]) -> Tuple[int, int, int]:
    dists = (label[dim] - point[dim] for dim in range(3))
    return tuple(map(lambda x: np.clip(x, -1, 1), dists))


def get_direction_to_label_unit_vector(point: Tuple[int, int, int], label: Tuple[int, int, int]) -> Tuple[int, int, int]:
    dists = [label[dim] - point[dim] for dim in range(3)]
    norm = sum([v**2 for v in dists])**0.5
    if norm == 0:
        return (0, 0, 0)
    return tuple(map(lambda x:x/norm, dists))


def create_image_label(image: np.ndarray, label: Tuple[int, int, int]) -> np.ndarray:
    empty_labels = create_empty_image_label(image, label)
    xdim, ydim, zdim, _ = empty_labels.shape
    for i in range(xdim):
        for j in range(ydim):
            for k in range(zdim):
                # TODO: add argument for label unit vector or discrete
                dirs = get_direction_to_label_unit_vector((i, j, k), label)
                empty_labels[i, j, k, :] = dirs
    return empty_labels
