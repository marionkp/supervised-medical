from typing import Tuple
import math

import torch


# TODO: set device type

# TODO: add some layer norms? other improvements to network?
class Net(torch.nn.Module):

    def __init__(self, roi_len: Tuple[int, int, int]):
        super().__init__()
        roi_size = tuple([2 * v - 1 for v in roi_len])
        out_chan = 5
        kernel_size = (3, 3, 3)
        self.conv3d = torch.nn.Conv3d(1, out_chan, kernel_size)
        conv_out_shape = math.prod([roi_size[i] - (kernel_size[i] // 2 + 1) for i in range(3)])
        hidden_size = (out_chan * conv_out_shape // 2) + 1
        self.fc1 = torch.nn.Linear(out_chan * conv_out_shape, hidden_size)
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
        x = torch.nn.functional.tanh(x)
        # TODO: add a unit vector normalising final activation
        return x
