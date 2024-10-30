from typing import Tuple

import logging
import numpy as np
import torch
import wandb

from src.env import eps_greedy_episode
from src.generate_dummy_data import random_eps_greedy_episode
from src.medical_loader import MedicalEnv
from src.nn_architecture import Net
from src.replay_buffer import ReplayBuffer
from src.utils import get_device


def batch_train(
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rb: ReplayBuffer,
    model: torch.nn.Module,
    batch_size: int,
) -> torch.nn.Module:
    roi, label = rb.sample_roi_and_label(batch_size)
    roi = torch.from_numpy(roi).to(dtype=torch.float32).to(get_device())
    pred_direction = model(roi)
    true_direction = torch.from_numpy(label).to(dtype=torch.float32).to(get_device())
    loss = criterion(pred_direction, true_direction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def warmup_replay_buffer(
    warmup_size: int, env: MedicalEnv, max_steps: int, roi_len: Tuple[int, int, int], rb: ReplayBuffer
):
    assert warmup_size <= rb.max_size
    epsilon = 1
    model = None  # No model prediction during model
    while len(rb) < warmup_size:
        image_data, image_label, landmark = env.sample_image_label_landmark()
        eps_greedy_episode(image_data, image_label, landmark, max_steps, epsilon, roi_len, model, rb)


# TODO: should I log the gradients? Separate them?
def get_grad_norm(model: torch.nn.Module) -> float:
    return np.sqrt(sum([torch.norm(p.grad) ** 2 for p in model.parameters()]))


def training_run():
    roi_len = (2, 2, 2)
    image_dims = (6, 6, 6)
    min_epsilon = 0
    epsilon = 1
    delta = 0.001
    max_steps = 100
    max_replay_size = 16  # 10000
    batch_size = 16
    lr = 1e-3
    landmark_index = 0

    warmup_size = max(batch_size, max_replay_size // 100)

    config = {
        "start_epsilon": epsilon,
        "delta": delta,
        "image_dims": image_dims,
        "roi_len": roi_len,
        "max_steps": max_steps,
        "max_replay_size": max_replay_size,
        "batch_size": batch_size,
        "warmup_size": warmup_size,
        "min_epsilon": min_epsilon,
        "learning_rate": lr,
        "landmark_index": landmark_index,
    }
    logging.info(f"{config=}")
    wandb.init(
        project="supervised-medical-mayou", config=config, mode="disabled"
    )  # TODO: add argparse for mode="disabled" or "online"

    # TODO: add dry-run/random data argparse option

    # TODO: move these hardcoded paths to argparse
    image_files = "/mnt/d/project_guy/filenames/image_files.txt"
    landmark_files = "/mnt/d/project_guy/filenames/landmark_files.txt"
    env = MedicalEnv(image_files, landmark_files, landmark_index)
    model = Net(roi_len).to(get_device())
    rb = ReplayBuffer(max_replay_size)
    # TODO: is this criterion correct?
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_replay_buffer(warmup_size, env, max_steps, roi_len, rb)
    for episode in range(100000):
        image_data, image_label, landmark = env.sample_image_label_landmark()
        steps = eps_greedy_episode(image_data, image_label, landmark, max_steps, epsilon, roi_len, model, rb)
        # TODO: run multiple batch trains per random greedy eps
        loss = batch_train(criterion, optimizer, rb, model, batch_size)
        log_dict = {
            "episode": episode,
            "loss": loss.item(),
            "epsilon": epsilon,
            "replay_buffer_size": len(rb),
            "steps": steps,
        }
        wandb.log(log_dict)
        logging.info(f"{log_dict=}")
        epsilon = max(min_epsilon, epsilon - delta)


if __name__ == "__main__":
    # TODO: add argparse here
    training_run()
