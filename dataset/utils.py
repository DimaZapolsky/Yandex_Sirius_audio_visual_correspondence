import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn


def transform(input, batch_format=False, log_base=21):
    x, y = input.shape[-2:]

    if batch_format:
        if len(input.shape) == 3:
            input = input[:, None, :, :]
    elif len(input.shape) == 3:
        input = input[None, :, :, :]
    else:
        input = input[None, None, :, :]

    xl = np.linspace(-1, 1, y)
    yl = np.linspace(-1, 1, y)

    xl = (np.power(log_base, (xl + 1) / 2) - (log_base // 2 + 1)) / (log_base // 2)

    xv, yv = np.meshgrid(xl, yl)

    x_out = torch.zeros(input.shape[:1] + (y, y, 2))
    x_out[:, :, :, 0] = torch.from_numpy(xv)
    x_out[:, :, :, 1] = torch.from_numpy(yv)

    return F.grid_sample(input, x_out).squeeze()


def get_picture_from_model_ans(model_answer, example_type, picture_shape, device):
    if example_type == 'pca':
        return get_pca_picture(model_answer, picture_shape, device)
    elif example_type == 'l1':
        return get_l1_picture(model_answer, picture_shape, device)
    elif example_type == 'l2':
        return get_l2_picture(model_answer, picture_shape, device)
    else:
        raise TypeError('unknow example type: {}'.format(example_type))


def get_pca_picture(model_answer, picture_shape, device):
    pca = sklearn.decomposition.PCA(n_components=3)
    vectors_square = model_answer[-1, :, :, :, :]
    vectors_square = vectors_square.reshape(vectors_square.shape[:-2] + (-1,))
    vectors_flatten = vectors_square.reshape(-1, vectors_square.shape[-1]).cpu().numpy()
    rgb = pca.fit_transform(vectors_flatten)
    rgb = rgb - np.min(rgb)
    rgb = rgb / np.max(rgb)

    rgb_picture = np.reshape(rgb, vectors_square.shape[:2] + (-1,))
    located_sound_picture = np.transpose(rgb_picture, [2, 0, 1])
    full_sound = torch.from_numpy(located_sound_picture).to(device)
    full_sound = full_sound[None, :, :, :]

    x_out = torch.zeros((1,) + picture_shape[1:] + (2,)).to(device)
    nx = np.linspace(-1, 1, picture_shape[1])
    ny = np.linspace(-1, 1, picture_shape[2])
    nxv, nyv = np.meshgrid(nx, ny)
    x_out[:, :, :, 0] = torch.from_numpy(nxv).to(device)
    x_out[:, :, :, 1] = torch.from_numpy(nyv).to(device)

    return F.grid_sample(full_sound, x_out)


def get_l1_picture(model_answer, picture_shape, device):
    

