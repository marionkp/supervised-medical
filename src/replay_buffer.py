from typing import Tuple
import random

import numpy as np


class ReplayBuffer:

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.index = 0
        self.rois = []
        self.labels = []

    def add_to_buffer(self, roi_and_label: Tuple[np.ndarray, np.ndarray]) -> None:
        roi, label = roi_and_label
        if len(self) < self.max_size:
            self.rois.append(roi)
            self.labels.append(label)
        else:
            self.rois[self.index] = roi
            self.labels[self.index] = label
        self.index = (self.index + 1) % self.max_size

    def sample_roi_and_label(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        assert batch_size <= len(self), f"batch size {batch_size} is larger than the replay buffer size {len(self)}"
        sampled_indices = random.sample(range(len(self)), k=batch_size)
        return (np.array(self.rois)[sampled_indices], np.array(self.labels)[sampled_indices])

    def sample_roi_and_label_fake(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        return (np.ones((batch_size, *self.rois[0].shape)), np.ones((batch_size, 3)))

    def __len__(self):
        return len(self.rois)
