import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch

import wandb
from src.env import eps_greedy_episode_batched
from src.medical_loader import MedicalEnv
from src.nn_architecture import Net
from src.replay_buffer import ReplayBuffer
from src.utils import agnostic_path, get_device
from src.logger import write_to_board, log
import tqdm


def batch_train(
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rb: ReplayBuffer,
    model: torch.nn.Module,
    batch_size: int,
) -> float:
    roi, label = rb.sample_roi_and_label(batch_size)
    roi = torch.from_numpy(roi).to(dtype=torch.float32, device=get_device())
    optimizer.zero_grad()
    pred_direction = model(roi)
    true_direction = torch.from_numpy(label).to(dtype=torch.float32, device=get_device())
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
    batch_size: int,
):
    assert warmup_size <= rb.max_size
    epsilon = 1
    model = None  # No model prediction during warmup
    log(f"Starting warmup for {warmup_size}")
    while len(rb) < warmup_size:
        image_data, image_label, landmark = env.sample_image_label_landmark(mode="train")
        eps_greedy_episode_batched(
            batch_size, image_data, image_label, landmark, max_steps, epsilon, roi_len, stride, model, rb
        )
    log(f"Finished warmup with {len(rb)=}")


# TODO: should I log the gradients? Separate them?
def get_grad_norm(model: torch.nn.Module) -> float:
    return np.sqrt(sum([torch.norm(p.grad) ** 2 for p in model.parameters()]))


def training_run():
    roi_len = (40, 40, 40)
    stride = 4
    min_epsilon = 0.1
    epsilon = 1
    delta = 0.01  # Increased from 0.001 to 0.01 for faster decay
    max_steps = 100
    max_replay_size = 10000
    collection_batch_size = 4
    train_batch_size = 256
    lr = 1e-3  # Increased from 1e-4 to 1e-3
    scheduler_t_max = 1000
    landmark_index = 0
    batch_trained_per_epoch = 4
    episodes_collected_per_epoch = 1
    debug_max_num_files = 1  # TODO: None
    debug_starting_position = None
    debug_image_type = "real"  # dummy or real
    debug_dummy_image_dims = (10, 10, 10)
    cache_images = True  # TODO: Set to false

    # at each epoch the number of samples trained on is:
    # batch_trained_per_epoch * train_batch_size

    # at each epoch the number of samples collected on is:
    # episodes_collected_per_epoch * collection_batch_size * max_size

    warmup_size = max(train_batch_size, max_replay_size // 100)

    model = Net(roi_len, stride).to(get_device())
    model_num_params = sum(p.numel() for p in model.parameters())

    config = {
        "start_epsilon": epsilon,
        "delta": delta,
        "roi_len": roi_len,
        "stride": stride,
        "max_steps": max_steps,
        "max_replay_size": max_replay_size,
        "train_batch_size": train_batch_size,
        "collection_batch_size": collection_batch_size,
        "warmup_size": warmup_size,
        "min_epsilon": min_epsilon,
        "learning_rate": lr,
        "landmark_index": landmark_index,
        "batch_trained_per_epoch": batch_trained_per_epoch,
        "episodes_collected_per_epoch": episodes_collected_per_epoch,
        "model_num_params": model_num_params,
        "debug_max_num_files": debug_max_num_files,
        "debug_starting_position": debug_starting_position,
        "debug_image_type": debug_image_type,
        "debug_dummy_image_dims": debug_dummy_image_dims,
        "scheduler_t_max": scheduler_t_max,
        "cache_images": cache_images,
    }
    log(f"{config=}")
    wandb_run = wandb.init(
        project="supervised-medical-mayou", config=config, mode="disabled"
    )  # TODO: add argparse for mode="disabled", "online" or "offline"

    # TODO: add dry-run/random data argparse option

    # TODO: move these hardcoded paths to argparse
    train_image_files = agnostic_path("/root/mayou/files/filenames/image_files_train.txt")
    train_landmark_files = agnostic_path("/root/mayou/files/filenames/landmark_files_train.txt")
    test_image_files = agnostic_path("/root/mayou/files/filenames/image_files_test.txt")
    test_landmark_files = agnostic_path("/root/mayou/files/filenames/landmark_files_test.txt")
    env = MedicalEnv(
        train_image_files,
        train_landmark_files,
        test_image_files,
        test_landmark_files,
        landmark_index,
        cache_images,
        debug_max_num_files,
        debug_image_type,
        debug_dummy_image_dims,
    )
    rb = ReplayBuffer(max_replay_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    warmup_replay_buffer(warmup_size, env, max_steps, roi_len, stride, rb, collection_batch_size)

    episode = 0
    best_loss = float("inf")
    while True:
        sampling_image_avg_time = 0
        eps_greedy_episode_avg_time = 0
        all_steps = []
        all_distances = []
        for _ in range(episodes_collected_per_epoch):
            sampling_image_start_time = time.time()
            image_data, image_label, landmark = env.sample_image_label_landmark(mode="train")
            sampling_image_avg_time += time.time() - sampling_image_start_time
            eps_greedy_episode_start_time = time.time()
            steps, final_dists = eps_greedy_episode_batched(
                collection_batch_size,
                image_data,
                image_label,
                landmark,
                max_steps,
                epsilon,
                roi_len,
                stride,
                model,
                rb,
            )

            all_steps += steps
            all_distances += final_dists
            eps_greedy_episode_avg_time += time.time() - eps_greedy_episode_start_time
        sampling_image_avg_time /= episodes_collected_per_epoch
        eps_greedy_episode_avg_time /= episodes_collected_per_epoch
        batch_train_start_time = time.time()
        loss_avg = 0
        for _ in range(batch_trained_per_epoch):
            loss_avg += batch_train(criterion, optimizer, scheduler, rb, model, train_batch_size)
        loss_avg /= batch_trained_per_epoch
        batch_train_time = time.time() - batch_train_start_time

        log_dict = {
            "train_loss": loss_avg,
            "epsilon": epsilon,
            "replay_buffer_size": len(rb),
            "train_steps_avg": sum(all_steps) / len(all_steps),
            "train_steps_min": min(all_steps),
            "train_steps_max": max(all_steps),
            "train_final_dist_avg": sum(all_distances) / len(all_distances),
            "train_final_dist_max": max(all_distances),
            "train_final_dist_min": min(all_distances),
            "sampling_image_time": sampling_image_avg_time,
            "eps_greedy_episode_time": eps_greedy_episode_avg_time,
            "batch_train_time": batch_train_time,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        write_to_board(log_dict, index=episode)
        if episode % 100 == 0:
            last_model_path = os.path.join(wandb_run.dir, "last.pt")
            torch.save(model, last_model_path)
        if episode % 10 == 0 and episode > 0:  # TODO: is that an appropriate evaluation period?
            log("Starting evaluation round")
            image_data, image_label, landmark = env.sample_image_label_landmark(mode="test")
            steps, final_dists = eps_greedy_episode_batched(
                collection_batch_size,
                image_data,
                image_label,
                landmark,
                max_steps,
                0,
                roi_len,
                stride,
                model,
                rb,
            )
            log_dict = {
                "test_steps_avg": sum(steps) / len(steps),
                "test_steps_min": min(steps),
                "test_steps_max": max(steps),
                "test_final_dist_avg": sum(final_dists) / len(final_dists),
                "test_final_dist_max": max(final_dists),
                "test_final_dist_min": min(final_dists),
            }
            write_to_board(log_dict, index=episode)
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
