import os
import numpy as np

import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agnostic_path(path: str) -> str:
    mount_txt = "/mnt/"
    if os.name == "nt" and path.startswith(mount_txt):
        linux_path = path[len(mount_txt) :]
        parts = [linux_path.split("/")[0].upper() + ":"] + linux_path.split("/")[1:]
        windows_path = "\\".join(parts)
        return windows_path
    return path
