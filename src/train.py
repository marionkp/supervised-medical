from typing import Optional, Tuple
import os
import time

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
    scheduler: torch.optim.lr_scheduler,
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
    scheduler.step()
    loss = loss.detach().cpu().item()
    return loss


def warmup_replay_buffer(
    warmup_size: int,
    env: MedicalEnv,
    max_steps: int,
    roi_len: Tuple[int, int, int],
    stride: int,
    rb: ReplayBuffer,
    debug_starting_position: Optional[Tuple[int, int, int]],
):
    assert warmup_size <= rb.max_size
    epsilon = 1
    model = None  # No model prediction during model
    while len(rb) < warmup_size:
        image_data, image_label, landmark = env.sample_image_label_landmark()
        eps_greedy_episode(
            image_data, image_label, landmark, max_steps, epsilon, roi_len, stride, model, rb, debug_starting_position
        )


# TODO: should I log the gradients? Separate them?
def get_grad_norm(model: torch.nn.Module) -> float:
    return np.sqrt(sum([torch.norm(p.grad) ** 2 for p in model.parameters()]))


def training_run():
    roi_len = (40, 40, 40)
    stride = 4
    min_epsilon = 0
    epsilon = 1
    delta = 0.001
    max_steps = 100
    max_replay_size = 4000  # 10000
    batch_size = 1024  # 4096
    # TODO: learning rate schedule? warmup and then decrease
    lr = 1e-4
    landmark_index = 0
    batch_trained_per_epoch = 1
    episodes_collected_per_epoch = 4
    debug_max_num_files = 1
    debug_starting_position = None
    debug_image_type = "real"  # dummy or real
    debug_dummy_image_dims = (10, 10, 10)

    warmup_size = max(batch_size, max_replay_size // 100)

    model = Net(roi_len, stride).to(get_device())
    model_num_params = sum(p.numel() for p in model.parameters())

    config = {
        "start_epsilon": epsilon,
        "delta": delta,
        "roi_len": roi_len,
        "stride": stride,
        "max_steps": max_steps,
        "max_replay_size": max_replay_size,
        "batch_size": batch_size,
        "warmup_size": warmup_size,
        "min_epsilon": min_epsilon,
        "learning_rate": lr,
        "landmark_index": landmark_index,
        "batch_trained_per_epoch": batch_trained_per_epoch,
        "model_num_params": model_num_params,
        "debug_max_num_files": debug_max_num_files,
        "debug_starting_position": debug_starting_position,
        "debug_image_type": debug_image_type,
        "debug_dummy_image_dims": debug_dummy_image_dims,
    }
    logging.info(f"{config=}")
    wandb_run = wandb.init(
        project="supervised-medical-mayou", config=config, mode="online"
    )  # TODO: add argparse for mode="disabled" or "online"

    # TODO: add dry-run/random data argparse option

    # TODO: move these hardcoded paths to argparse
    image_files = "/mnt/d/project_guy/filenames/image_files.txt"
    landmark_files = "/mnt/d/project_guy/filenames/landmark_files.txt"
    env = MedicalEnv(
        image_files, landmark_files, landmark_index, debug_max_num_files, debug_image_type, debug_dummy_image_dims
    )
    rb = ReplayBuffer(max_replay_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # TODO: find a more principled tmax?
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * batch_trained_per_epoch)
    warmup_replay_buffer(warmup_size, env, max_steps, roi_len, stride, rb, debug_starting_position)
    episode = 0
    best_loss = float("inf")
    while True:
        sampling_image_avg_time = 0
        eps_greedy_episode_avg_time = 0
        steps_avg = 0
        final_dist_avg = 0
        for _ in range(episodes_collected_per_epoch):
            sampling_image_start_time = time.time()
            image_data, image_label, landmark = env.sample_image_label_landmark()
            sampling_image_avg_time += time.time() - sampling_image_start_time
            eps_greedy_episode_start_time = time.time()
            steps, final_dist = eps_greedy_episode(
                image_data,
                image_label,
                landmark,
                max_steps,
                epsilon,
                roi_len,
                stride,
                model,
                rb,
                debug_starting_position,
            )
            steps_avg += steps
            final_dist_avg += final_dist
            eps_greedy_episode_avg_time += time.time() - eps_greedy_episode_start_time
        sampling_image_avg_time /= episodes_collected_per_epoch
        eps_greedy_episode_avg_time /= episodes_collected_per_epoch
        final_dist_avg /= episodes_collected_per_epoch
        steps_avg /= episodes_collected_per_epoch
        batch_train_start_time = time.time()
        loss_avg = 0
        for _ in range(batch_trained_per_epoch):
            loss_avg += batch_train(criterion, optimizer, scheduler, rb, model, batch_size)
        loss_avg /= batch_trained_per_epoch
        batch_train_time = time.time() - batch_train_start_time
        log_dict = {
            "episode": episode,
            "loss": loss_avg,
            "epsilon": epsilon,
            "replay_buffer_size": len(rb),
            "steps": steps_avg,
            "final_dist": final_dist_avg,
            "sampling_image_time": sampling_image_avg_time,
            "eps_greedy_episode_time": eps_greedy_episode_avg_time,
            "batch_train_time": batch_train_time,
        }
        wandb.log(log_dict)
        logging.info(f"{log_dict=}")
        if episode % 100 == 0:
            last_model_path = os.path.join(wandb_run.dir, "last.pt")
            torch.save(model, last_model_path)
        # TODO: better best model saving strategy based on evaluation data?
        if best_loss > loss_avg:
            best_loss = loss_avg
            best_model_path = os.path.join(wandb_run.dir, "best.pt")
            torch.save(model, best_model_path)
        epsilon = max(min_epsilon, epsilon - delta)
        episode += 1


if __name__ == "__main__":
    # TODO: add argparse here
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    training_run()
