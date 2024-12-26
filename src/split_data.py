from typing import List

import numpy as np

from src.medical_loader import read_paths_from_file
from src.utils import agnostic_path


def main(split_ratio: float, all_landmark_path: str, all_image_path: str) -> None:
    assert 0 <= split_ratio <= 1
    all_landmarks = read_paths_from_file(all_landmark_path)
    all_images = read_paths_from_file(all_image_path)
    assert len(all_landmarks) == len(all_images)
    training_size = round(split_ratio * len(all_landmarks))
    training_indices = np.random.choice(list(range(len(all_landmarks))), size=training_size, replace=False)
    test_indices = [i for i in range(len(all_landmarks)) if i not in set(training_indices)]

    def write_to_file(file: str, indices: List[int], paths: List[str]) -> None:
        with open(file, "w") as f:
            for index in indices:
                f.write(f"{paths[index]}\n")

    train_image_files = agnostic_path("/mnt/d/project_guy/filenames/image_files_train.txt")
    train_landmark_files = agnostic_path("/mnt/d/project_guy/filenames/landmark_files_train.txt")
    test_image_files = agnostic_path("/mnt/d/project_guy/filenames/image_files_test.txt")
    test_landmark_files = agnostic_path("/mnt/d/project_guy/filenames/landmark_files_test.txt")
    write_to_file(train_image_files, training_indices, all_images)
    write_to_file(train_landmark_files, training_indices, all_landmarks)
    write_to_file(test_image_files, test_indices, all_images)
    write_to_file(test_landmark_files, test_indices, all_landmarks)


if __name__ == "__main__":
    image_files = agnostic_path("/mnt/d/project_guy/filenames/image_files.txt")
    landmark_files = agnostic_path("/mnt/d/project_guy/filenames/landmark_files.txt")
    split_ratio = 0.8
    main(split_ratio, landmark_files, image_files)
