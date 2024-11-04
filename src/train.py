from typing import Optional, Tuple

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
) -> float:
    roi, label = rb.sample_roi_and_label(batch_size)
    roi = torch.from_numpy(roi).to(dtype=torch.float32).to(get_device())
    optimizer.zero_grad()
    pred_direction = model(roi)
    true_direction = torch.from_numpy(label).to(dtype=torch.float32).to(get_device())
    loss = criterion(pred_direction, true_direction)
    loss.backward()
    optimizer.step()
    loss = loss.detach().cpu().item()
    return loss

def warmup_replay_buffer(
    warmup_size: int, env: MedicalEnv, max_steps: int, roi_len: Tuple[int, int, int], rb: ReplayBuffer, debug_starting_position: Optional[Tuple[int, int, int]]
):
    assert warmup_size <= rb.max_size
    epsilon = 1
    model = None  # No model prediction during model
    while len(rb) < warmup_size:
        image_data, image_label, landmark = env.sample_image_label_landmark()
        eps_greedy_episode(image_data, image_label, landmark, max_steps, epsilon, roi_len, model, rb, debug_starting_position)


# TODO: should I log the gradients? Separate them?
def get_grad_norm(model: torch.nn.Module) -> float:
    return np.sqrt(sum([torch.norm(p.grad) ** 2 for p in model.parameters()]))


def training_run():
    roi_len = (8, 8, 8)
    min_epsilon = 0
    epsilon = 1
    delta = 0.001
    max_steps = 100
    max_replay_size = 5000
    batch_size = 32
    lr = 1e-3
    landmark_index = 0
    batch_trained_per_episode = 10
    debug_max_num_files = 1
    debug_starting_position = None
    warmup_size = max(batch_size, max_replay_size // 100)

    config = {
        "start_epsilon": epsilon,
        "delta": delta,
        "roi_len": roi_len,
        "max_steps": max_steps,
        "max_replay_size": max_replay_size,
        "batch_size": batch_size,
        "warmup_size": warmup_size,
        "min_epsilon": min_epsilon,
        "learning_rate": lr,
        "landmark_index": landmark_index,
        "batch_trained_per_episode": batch_trained_per_episode,
        "debug_max_num_files": debug_max_num_files,
        "debug_starting_position": debug_starting_position,
    }
    logging.info(f"{config=}")
    wandb.init(
        project="supervised-medical-mayou", config=config, mode="online"
    )  # TODO: add argparse for mode="disabled" or "online"

    # TODO: add dry-run/random data argparse option

    # TODO: move these hardcoded paths to argparse
    image_files = "/mnt/d/project_guy/filenames/image_files.txt"
    landmark_files = "/mnt/d/project_guy/filenames/landmark_files.txt"
    env = MedicalEnv(image_files, landmark_files, landmark_index, debug_max_num_files)
    model = Net(roi_len).to(get_device())
    rb = ReplayBuffer(max_replay_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_replay_buffer(warmup_size, env, max_steps, roi_len, rb, debug_starting_position)
    episode = 0
    while True:
        image_data, image_label, landmark = env.sample_image_label_landmark()
        steps = eps_greedy_episode(image_data, image_label, landmark, max_steps, epsilon, roi_len, model, rb, debug_starting_position)
        total_loss = 0
        for _ in range(batch_trained_per_episode):
            total_loss += batch_train(criterion, optimizer, rb, model, batch_size)
        loss = total_loss / batch_trained_per_episode
        log_dict = {
            "episode": episode,
            "loss": loss,
            "epsilon": epsilon,
            "replay_buffer_size": len(rb),
            "steps": steps,
        }
        wandb.log(log_dict)
        logging.info(f"{log_dict=}")
        # TODO: save model every once and a while
        epsilon = max(min_epsilon, epsilon - delta)
        episode += 1


if __name__ == "__main__":
    # TODO: add argparse here
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    training_run()
