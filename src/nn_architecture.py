import math
from typing import Tuple

import torch

from src.env import get_roi_dims_from_len_and_stride

# TODO: set device type


# TODO: add some layer norms? other improvements to network?
class Net(torch.nn.Module):
    def __init__(self, roi_len: Tuple[int, int, int], stride: int):
        super().__init__()
        roi_size = get_roi_dims_from_len_and_stride(roi_len, stride)
        out_chan = 4
        kernel_size = (16, 16, 16)
        # TODO: any custom weight initialisation?
        self.conv3d = torch.nn.Conv3d(1, out_chan, kernel_size)
        # 3d Conv Shape description: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d
        padding = 0
        dilation = 1
        stride = 1
        conv3d_out_shape = [
            int((roi_size[i] + 2 * padding - dilation * (kernel_size[i] - 1) / stride - 1) + 1) for i in range(3)
        ]
        conv_out_shape = int(math.prod((out_chan, *conv3d_out_shape)))
        hidden_size = conv_out_shape // 2 + 1
        self.fc1 = torch.nn.Linear(conv_out_shape, hidden_size)
        self.output = torch.nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:  # add batch of size 1
            x = x.unsqueeze(0)
        if len(x.shape) == 4:  # add channel in second axis of size 1
            x = x.unsqueeze(1)
        x = self.conv3d(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.leaky_relu(x)
        x = self.fc1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.output(x)
        # NOTE: Tanh output layer prevents training, how come?
        # TODO: add a unit vector normalising final activation
        return x
