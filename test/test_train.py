import numpy as np
import torch

from src.env import get_roi_dims_from_len_and_stride
from src.nn_architecture import Net
from src.replay_buffer import ReplayBuffer
from src.train import batch_train
from src.utils import get_device


def test_batch_independence():
    # Test batch of size 1, no batch, and batch of size 2 are the same
    pass


def test_same_position_training():
    roi_len = (5, 5, 5)
    stride = 2
    roi_dims = get_roi_dims_from_len_and_stride(roi_len, stride)
    batch_size = 1
    lr = 1e-3
    model = Net(roi_len, stride).to(get_device())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rb = ReplayBuffer(10)
    roi = np.ones(roi_dims)
    direction_label = np.array([-1, 0, 1])
    rb.add_to_buffer((roi, direction_label))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    loss = batch_train(criterion, optimizer, scheduler, rb, model, batch_size)
    assert loss > 1e-2
    for _ in range(100):
        loss = batch_train(criterion, optimizer, scheduler, rb, model, batch_size)
    assert loss < 1e-5
