import math
from typing import Tuple

import torch
import torch.nn as nn

from src.env import get_roi_dims_from_len_and_stride


class Net(nn.Module):
    def __init__(self, roi_len: Tuple[int, int, int], stride: int):
        super().__init__()
        roi_size = get_roi_dims_from_len_and_stride(roi_len, stride)

        # Multiple conv layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(32, 32, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=4, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool3d(2)

        # Calculate flattened size
        self.flat_size = self._get_flat_size(roi_size)

        self.fc1 = nn.Linear(self.flat_size, 512)
        self.prelu4 = nn.PReLU()
        self.fc2 = nn.Linear(512, 256)
        self.prelu5 = nn.PReLU()
        self.fc3 = nn.Linear(256, 3)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_flat_size(self, input_size):
        # Helper function to calculate flattened size
        x = torch.randn(1, 1, *input_size)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        return int(torch.prod(torch.tensor(x.size())[1:]))

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:  # add batch of size 1
            x = x.unsqueeze(0)
        if len(x.shape) == 4:  # add channel in second axis of size 1
            x = x.unsqueeze(1)

        x = self.pool1(self.prelu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.prelu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.prelu3(self.bn3(self.conv3(x))))

        x = x.view(-1, self.flat_size)
        x = self.prelu4(self.fc1(x))
        x = self.prelu5(self.fc2(x))
        x = self.fc3(x)

        # Normalize direction vector
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        return x
