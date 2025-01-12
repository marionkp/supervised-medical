import os

from ray import get
from torch.utils.tensorboard.writer import SummaryWriter
import wandb
import logging

sw = None


def get_summary_writer() -> SummaryWriter:
    global sw
    if sw is None:
        comment = ""
        sw = SummaryWriter(comment)
        log(f"Logs to {sw.log_dir}")
    return sw


def log(message: str) -> None:
    sw = get_summary_writer()
    logging.info(str(message))
    with open(os.path.join(sw.log_dir, "logs.txt"), "a") as logs:
        logs.write(str(message) + "\n")


def write_to_board(scalars: dict, index: int = 0) -> None:
    sw = get_summary_writer()
    log(str(scalars))
    for key, value in scalars.items():
        sw.add_scalar(key, value, index)
    wandb.log({"episode": index, **scalars})
