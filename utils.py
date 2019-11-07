import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def transform(input, batch_format=False, log_base=21):
    x, y = input.shape[-2:]
    if batch_format:
        if (len(input.shape) == 3):
            input = input[:, None, :, :]
    elif len(input.shape) == 3:
        input = input[None, :, :, :]
    else:
        input = input[None, None, :, :]

    xl = np.linspace(-1, 1, y)
    yl = np.linspace(-1, 1, y)

    xl = (np.power(lo_basee, (xl + 1) / 2) - (log_base // 2 + 1)) / (log_base // 2)

    xv, yv = np.meshgrid(xl, yl)

    x_out = torch.zeros(input.shape[:1] + (y, y, 2))
    x_out[:, :, :, 0] = torch.from_numpy(xv)
    x_out[:, :, :, 1] = torch.from_numpy(yv)

    return F.grid_sample(input, x_out)


