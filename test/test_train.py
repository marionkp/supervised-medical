import numpy as np
import torch

from src.env import get_roi_dims_from_len_and_stride
from src.nn_architecture import Net
from src.replay_buffer import ReplayBuffer
from src.train import batch_train
from src.utils import get_device


def test_same_position_training():
    roi_len = (30, 30, 30)
    stride = 2
    roi_dims = get_roi_dims_from_len_and_stride(roi_len, stride)
    batch_size = 1
    lr = 1e-3
    model = Net(roi_len, stride).to(get_device())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rb = ReplayBuffer(10)
    roi = np.ones((1, *roi_dims))  # Add batch dimension
    direction_label = np.array([[-1, 0, 1]])  # Add batch dimension
    direction_label = direction_label / np.sqrt(np.sum(direction_label**2, axis=-1, keepdims=True))
    rb.add_to_buffer((roi, direction_label))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    loss = batch_train(criterion, optimizer, scheduler, rb, model, batch_size)
    assert loss > 1e-2
    for _ in range(100):
        loss = batch_train(criterion, optimizer, scheduler, rb, model, batch_size)
    assert loss < 1e-5
