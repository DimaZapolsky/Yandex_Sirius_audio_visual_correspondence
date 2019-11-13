import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn

from librosa.core import istft

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

    xv, yv = np.meshgrid(xl, yl, indexing='ij')

    x_out = torch.zeros(input.shape[:1] + (y, y, 2))
    x_out[:, :, :, 0] = torch.from_numpy(xv)
    x_out[:, :, :, 1] = torch.from_numpy(yv)

    return F.grid_sample(input, x_out).squeeze()


def get_picture_from_model_ans(model_answer, example_type, picture_shape, device):
    if example_type == 'pca':
        return get_pca_picture(model_answer, picture_shape, device)
    elif example_type == 'l1':
        return get_picture(model_answer, picture_shape, device)
    elif example_type == 'l2':
        return get_picture(model_answer, picture_shape, device, type_lin='l2')
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


def get_picture(model_answer, picture_shape, device, original_height=512, log_base=21, hop_length=256, window_lenght=1022, type_lin='l1'):
    input_tmp = model_answer[-1:, :].reshape((1, -1) + model_answer.shape[-2:])

    x_out = torch.zeros(input_tmp.shape[0], original_height, model_answer.shape[-1], 2)
    xl = np.linspace(-1, 1, original_height)
    yl = np.linspace(-1, 1, model_answer.shape[-1])

    xl = np.log((xl * (log_base // 2) + (log_base // 2 + 1))) / np.log(log_base) * 2 - 1
    nxv, nyv = np.meshgrid(xl, yl, indexing='ij')

    x_out[:, :, :, 0] = torch.from_numpy(nxv)
    x_out[:, :, :, 1] = torch.from_numpy(nyv)

    x_out = x_out.to(device)

    delog = F.grid_sample(input_tmp, x_out)
    delog = delog.reshape((-1,) + delog.shape[-2:])
    tmpl = []

    for i in range(delog.shape[0]):
        tmpl.append(istft(delog[i].cpu().numpy(), hop_length, window_lenght))
    delog = (np.array(tmpl))
    delog = delog.reshape((1,) + model_answer.shape[1:3] + (-1,))

    if (type_lin == 'l1'):
        delog = np.abs(delog)
        delog = np.mean(delog, axis=3)
    else:
        count = delog.shape[-1]
        delog = np.power(delog, 2)
        delog = np.sum(delog, axis=3)
        delog = np.power(delog, 0.5)
        delog /= count

    delog /= np.max(delog)

    delog = delog.squeeze(0)
    delog = np.stack((delog, delog, delog), axis=0)

    delog = np.repeat(delog, 16, axis=1)
    delog = np.repeat(delog, 16, axis=2)

    #image_low_res = torch.from_numpy(delog).to(device)[None, :]

    #x_out = torch.zeros((1,) + picture_shape[1:] + (2,)).to(device)
    #nx = np.linspace(-1, 1, picture_shape[1])
    #ny = np.linspace(-1, 1, picture_shape[2])
    #nxv, nyv = np.meshgrid(nx, ny)
    #x_out[:, :, :, 0] = torch.from_numpy(nxv).to(device)
    #x_out[:, :, :, 1] = torch.from_numpy(nyv).to(device)

    return torch.from_numpy(delog).to(device)


