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


def get_random_3d_pos(image_dims: Tuple[int, int, int], batch_size: int = None) -> np.ndarray:
    size = (3,)
    if batch_size is not None:
        size = (batch_size, 3)
    return np.random.randint(low=(0, 0, 0), high=image_dims, size=size)


def step_env(
    cur_pos: Tuple[int, int, int], direction: Tuple[int, int, int], image_shape: Tuple[int, int, int]
) -> np.ndarray:
    return np.clip(cur_pos + direction, 0, tuple(v - 1 for v in image_shape))


def dist_3d_points(start_points: np.ndarray, end_points: np.ndarray) -> float | np.ndarray:
    return np.linalg.norm(end_points - start_points, axis=-1)


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
) -> Tuple[int, float]:
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
        direction = get_eps_greedy_directions(np.array([roi]), epsilon, model)
        position = step_env(position, direction, image_data.shape)
        steps += 1


def eps_greedy_episode_batched(
    batch_size: int,
    image_data: np.ndarray,
    image_label: np.ndarray,
    landmark: Tuple[Tuple[int, int, int]],
    max_steps: int,
    epsilon: float,
    roi_len: Tuple[int, int, int],
    stride: int,
    model: torch.nn.Module,
    rb: ReplayBuffer,
) -> Tuple[float, float]:
    positions = get_random_3d_pos(image_data.shape, batch_size)
    steps = np.array([0] * batch_size)
    dones = np.array([False] * batch_size)
    while True:
        rois = []
        for i in range(batch_size):  # TODO: vectorise this
            pos = tuple(positions[i].tolist())
            if dones[i] or landmark == pos:
                dones[i] = True
                continue
            steps[i] += 1
            roi = get_roi_from_image(pos, image_data, roi_len, stride)
            direction_label = image_label[pos]
            rb.add_to_buffer((roi, direction_label))
            rois.append(roi)
        if (steps >= max_steps).any() or dones.all():
            final_dists = dist_3d_points(positions, landmark)
            return steps.mean(), final_dists.mean()
        rois = np.stack(rois, axis=0)
        not_dones = np.invert(dones)
        directions = get_eps_greedy_directions(rois, epsilon, model)
        positions[not_dones] = step_env(
            positions[not_dones], directions, image_data.shape
        )  # TODO: test and make sure that "done" agents stop moving


def get_eps_greedy_directions(rois: np.ndarray, epsilon: float, model: torch.nn.Module) -> np.ndarray:
    assert 0 <= epsilon <= 1
    if random.random() < epsilon:
        directions = get_random_directions(len(rois))
    else:
        directions = get_model_pred_direction(rois, model)
    return directions


def get_random_directions(batch_size: int) -> np.ndarray:
    return np.random.choice([-1, 0, 1], size=(batch_size, 3), replace=True)


def get_model_pred_direction(rois: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    roi = torch.from_numpy(rois).to(dtype=torch.float32).to(get_device())
    pred_directions = model(roi).squeeze()
    directions = pred_to_direction(pred_directions)
    return directions


def pred_to_direction(pred_direction: torch.Tensor) -> np.ndarray:
    x = pred_direction
    x[x >= 0.5] = 1
    x[x <= -0.5] = -1
    x[torch.logical_and(x > -0.5, x < 0.5)] = 0
    return x.detach().cpu().numpy()
