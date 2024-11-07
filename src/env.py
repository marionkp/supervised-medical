from typing import Optional, Tuple
import math
import random

import numpy as np
import torch

from src.replay_buffer import ReplayBuffer
from src.utils import get_device


def get_roi_dims_from_len_and_stride(roi_len: Tuple[int, int, int], stride: int) -> Tuple[int, int, int]:
    return tuple([math.ceil((2 * v - 1) / stride) for v in roi_len])


def get_roi_from_image(
    position: Tuple[int, int, int], image: np.ndarray, roi_len: Tuple[int, int, int], stride: int
) -> np.ndarray:
    """
    RoI len is the distance for one axis from the position (included) to the edge (size 2x-1).
    E.g. RoI dim (2, 1, 4) would return a 3D array of size (3, 1, 9)
    0s are padded if the RoI is outside of the image.
    """
    xpos, ypos, zpos = position
    xlen, ylen, zlen = roi_len
    assert xlen > 0 and ylen > 0 and zlen > 0
    xdim, ydim, zdim = image.shape
    roi = image[
        max(0, xpos - xlen + 1) : min(xpos + xlen, xdim),
        max(0, ypos - ylen + 1) : min(ypos + ylen, ydim),
        max(0, zpos - zlen + 1) : min(zpos + zlen, zdim),
    ]
    # TODO: is 0 a good value to pad? What is the imaging null value?
    padding = (
        (max(0, xlen - xpos - 1), max(0, xpos + xlen - xdim)),
        (max(0, ylen - ypos - 1), max(0, ypos + ylen - ydim)),
        (max(0, zlen - zpos - 1), max(0, zpos + zlen - zdim)),
    )
    roi = np.pad(roi, padding)[::stride, ::stride, ::stride]
    return roi


def get_random_3d_pos(image_dims: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple([np.random.randint(image_dims[i]) for i in range(3)])


def step_env(
    cur_pos: Tuple[int, int, int], direction: Tuple[int, int, int], image_shape: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    new_pos = tuple([np.clip(cur_pos[i] + direction[i], 0, image_shape[i] - 1) for i in range(3)])
    return new_pos


def dist_3d_points(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return sum([(i - j) ** 2 for i, j in zip(a, b)]) ** 0.5


def eps_greedy_episode(
    image_data: np.ndarray,
    image_label: np.ndarray,
    landmark: Tuple[int, int, int],
    max_steps: int,
    epsilon: float,
    roi_len: Tuple[int, int, int],
    stride: int,
    model: torch.nn.Module,
    rb: ReplayBuffer,
    debug_starting_position: Optional[Tuple[int, int, int]],
) -> int:
    if debug_starting_position is None:
        position = get_random_3d_pos(image_data.shape)
    else:
        position = debug_starting_position
    steps = 1
    while True:
        roi = get_roi_from_image(position, image_data, roi_len, stride)
        direction_label = image_label[position]
        rb.add_to_buffer((roi, direction_label))
        if position == landmark or steps >= max_steps:
            final_dist = dist_3d_points(position, landmark)
            return steps, final_dist
        direction = get_eps_greedy_direction(roi, epsilon, model)
        position = step_env(position, direction, image_data.shape)
        steps += 1


def get_eps_greedy_direction(roi: np.ndarray, epsilon: float, model: torch.nn.Module) -> Tuple[int, int, int]:
    assert 0 <= epsilon <= 1
    if random.random() < epsilon:
        direction = get_random_direction()
    else:
        direction = get_model_pred_direction(roi, model)
    return direction


def get_random_direction() -> Tuple[int, int, int]:
    return tuple(np.random.choice([-1, 0, 1], size=3, replace=True).tolist())


def get_model_pred_direction(roi: np.ndarray, model: torch.nn.Module) -> Tuple[int, int, int]:
    roi = torch.from_numpy(roi).to(dtype=torch.float32).to(get_device())
    pred_direction = model(roi).squeeze()
    direction = pred_to_direction(pred_direction)
    return direction


def pred_to_direction(pred_direction: torch.Tensor) -> Tuple[int, int, int]:
    def clip_dir(x: float) -> int:
        if x > 0.5:
            return 1
        if x < -0.5:
            return -1
        return 0

    return tuple([clip_dir(v) for v in pred_direction])
