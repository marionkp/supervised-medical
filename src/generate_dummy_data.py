from typing import Tuple

import numpy as np
import torch

from src.env import eps_greedy_episode, get_random_3d_pos
from src.replay_buffer import ReplayBuffer


# TODO: add function to generate dummy .nii.gz data


def generate_data(xdim: int, ydim: int, zdim: int, xlabel: int, ylabel: int, zlabel: int) -> np.ndarray:
    assert 0 <= xlabel < xdim
    assert 0 <= ylabel < ydim
    assert 0 <= zlabel < zdim
    data = np.zeros((xdim, ydim, zdim))
    longest_dist_to_edge = max(xdim - xlabel, xlabel, ydim - ylabel, ylabel, zdim - zlabel, zlabel)
    dummy_values = [0, 0.5, 1]
    for i in reversed(range(longest_dist_to_edge + 1)):
        val = dummy_values[i % len(dummy_values)]
        data[
            max(0, xlabel - i) : min(xdim, xlabel + i + 1),
            max(0, ylabel - i) : min(ydim, ylabel + i + 1),
            max(0, zlabel - i) : min(zdim, zlabel + i + 1),
        ] = val
    assert data.shape == (xdim, ydim, zdim)
    return data


# TODO: just a one where the landmark is
# def generate_data(xdim: int, ydim: int, zdim: int, xlabel: int, ylabel: int, zlabel: int) -> np.ndarray:
#     assert 0 <= xlabel < xdim
#     assert 0 <= ylabel < ydim
#     assert 0 <= zlabel < zdim
#     data = np.zeros((xdim, ydim, zdim))
#     data[xlabel, ylabel, zlabel] = 1
#     assert data.shape == (xdim, ydim, zdim)
#     return data


def random_eps_greedy_episode(
    image_dims: Tuple[int, int, int],
    max_steps: int,
    epsilon: float,
    roi_len: Tuple[int, int, int],
    stride: int,
    model: torch.nn.Module,
    rb: ReplayBuffer,
    keep_landmark_fixed: bool = False,
) -> int:
    if keep_landmark_fixed:
        landmark = (0, 0, 0)
    else:
        landmark = get_random_3d_pos(image_dims)
    image_data = generate_data(*image_dims, *landmark)
    steps, _ = eps_greedy_episode(image_data, landmark, max_steps, epsilon, roi_len, stride, model, rb)
    return steps
