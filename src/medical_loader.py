import logging
import random
import time
from typing import List, Optional, Tuple, Literal

import numpy as np
import SimpleITK as sitk

import wandb
from src.generate_dummy_data import generate_data
from src.generate_label import create_image_label
from src.utils import agnostic_path


def load_image(image_path: str) -> np.ndarray:
    image_path = agnostic_path(image_path)
    sitk_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    np_image = sitk.GetArrayFromImage(sitk_image)
    # threshold image between p10 and p99 then re-scale [0-255]
    p0 = np_image.min().astype("float")
    p10 = np.percentile(np_image, 10)
    p99 = np.percentile(np_image, 99)
    p100 = np_image.max().astype("float")
    sitk_image = sitk.Threshold(sitk_image, lower=p10, upper=p100, outsideValue=p10)
    sitk_image = sitk.Threshold(sitk_image, lower=p0, upper=p99, outsideValue=p99)
    sitk_image = sitk.RescaleIntensity(sitk_image, outputMinimum=0, outputMaximum=255)
    # Convert from [depth, width, height] to [width, height, depth]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(2, 1, 0)
    return image_data


def read_paths_from_file(file_path: str | None) -> List[str]:
    if file_path is None or file_path == "":
        return []
    with open(file_path, "r") as f:
        s = f.read().strip()
    paths = s.split("\n")
    paths = list(map(agnostic_path, paths))
    return paths


def read_landmark_file(file_path: str) -> Tuple[Tuple[int, int, int]]:
    """
    Example content of a landmark file:
    72, 81, 95
    72, 76, 98
    72, 89, 83
    72, 77, 87
    Each row represents the xyz coordinates of a landmark
    """
    with open(file_path, "r") as f:
        s = f.read()
    return str_to_landmarks(s)


def str_to_landmarks(landmark_str: str) -> Tuple[Tuple[int, int, int]]:
    s = landmark_str.strip().split("\n")
    res = tuple([tuple(map(int, v.split(","))) for v in s])
    return res


class MedicalEnv:
    def __init__(
        self,
        path_to_train_image_files: str,
        path_to_train_landmark_files: str,
        path_to_test_image_files: str | None,
        path_to_test_landmark_files: str | None,
        landmark_index: int,
        cache_images: bool,
        debug_max_num_files: Optional[int],
        debug_image_type: str,
        debug_dummy_image_dims: Optional[Tuple[int, int, int]],
    ):
        self.train_image_file_paths = read_paths_from_file(path_to_train_image_files)
        self.train_landmark_file_paths = read_paths_from_file(path_to_train_landmark_files)
        self.test_image_file_paths = read_paths_from_file(path_to_test_image_files)
        self.test_landmark_file_paths = read_paths_from_file(path_to_test_landmark_files)
        assert len(self.train_image_file_paths) == len(self.train_landmark_file_paths) and len(
            self.test_image_file_paths
        ) == len(self.test_landmark_file_paths)
        self.num_train_files = len(self.train_image_file_paths)
        self.num_test_files = len(self.test_image_file_paths)
        if debug_max_num_files is not None:
            self.num_train_files = min(debug_max_num_files, self.num_train_files)
        self.landmark_index = landmark_index
        self.path_to_data = {}
        self.cache_images = cache_images
        self.debug_image_type = debug_image_type
        self.debug_dummy_image_dims = debug_dummy_image_dims

    def get_image_label_landmark(
        self, index: int, mode: Literal["train", "test"]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        image_load_start_time = time.time()
        if index not in self.path_to_data:
            if mode == "train":
                image_data = load_image(self.train_image_file_paths[index])
                landmark = read_landmark_file(self.train_landmark_file_paths[index])[self.landmark_index]
            elif mode == "test":
                image_data = load_image(self.test_image_file_paths[index])
                landmark = read_landmark_file(self.test_landmark_file_paths[index])[self.landmark_index]
            else:
                raise NotImplementedError()
            label = create_image_label(image_data, landmark)
            logging.info(f"Loaded image and labels at index {index} - image shape {image_data.shape}")
            res = image_data, label, landmark
        else:
            logging.debug(f"Retrieving image and labels from cache at index {index}")
            res = self.path_to_data[index]
        if self.cache_images:
            self.path_to_data[index] = res
        image_load_time = time.time() - image_load_start_time
        wandb.log({"image_loading_time": image_load_time})  # TODO: use write_to_board for this as well?
        return res

    def sample_image_label_landmark(
        self, mode: Literal["train", "test"]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        if self.debug_image_type == "real":
            sampled_index = self.sample_file_index(mode)
            return self.get_image_label_landmark(sampled_index, mode)
        if self.debug_image_type == "dummy":
            return self.sample_dummy_image_label_landmark()
        raise NotImplementedError()

    def sample_file_index(self, mode: Literal["train", "test"]) -> int:
        if mode == "train":
            return random.randint(0, self.num_train_files - 1)
        if mode == "test":
            return random.randint(0, self.num_test_files - 1)
        raise NotImplementedError()

    def sample_dummy_image_label_landmark(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        assert self.debug_dummy_image_dims is not None, f"debug_dummy_image_dims is None, provide some dummy dimensions"
        landmark = get_random_3d_pos(self.debug_dummy_image_dims)
        image_data = generate_data(*self.debug_dummy_image_dims, *landmark)
        label = create_image_label(image_data, landmark)
        return (image_data, label, landmark)
