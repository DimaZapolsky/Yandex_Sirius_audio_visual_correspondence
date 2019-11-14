import models
import numpy as np
import sklearn.decomposition
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from models import *
from dataset import Dataset
import time
import adabound
from audio import base
from audio.base import Unet
from librosa.core import istft
from dataset.utils import get_picture_from_model_ans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', required=True, type=int, help='Video\'s fps')
    parser.add_argument('--width', required=True, type=int, help='Image width')
    parser.add_argument('--height', required=True, type=int, help='Image height')
    parser.add_argument('--freq', required=True, type=int, help='Audio frequency')
    parser.add_argument('--train-set-dir', default='./', help='Train dataset directory')
    parser.add_argument('--dev-set-dir', default='./', help='Dev dataset directory')
    parser.add_argument('--test-set-dir', default='./', help='Test dataset directory')
    parser.add_argument('--window-len', default=1022, type=int, help='Window length in STFT')
    parser.add_argument('--overlap-len', default=766, type=int, help='Window overlap in STFT')
    parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
    parser.add_argument('--N', default=2, type=int, help='Count of mixed videos')
    parser.add_argument('--K', default=16, type=int, help='Count of video features per pixel')
    parser.add_argument('--fragment-len', default=6, type=float, help='Durability of video/audio fragments')
    parser.add_argument('--video-model-lr', default=1e-4, type=float, help='Learning rate for video model')
    parser.add_argument('--audio-model-lr', default=1e-3, type=float, help='Learning rate for audio model')
    parser.add_argument('--generator-lr', default=1e-3, type=float, help='Learning rate for generator')
    parser.add_argument('--epoch-loss-freq', default=50, type=int, help='Number of epochs to print losses')
    parser.add_argument('--save-model-freq', default=300, type=int, help='Number of batches to save model')
    parser.add_argument('--example-freq', default=1, type=int, help='Batches to print example')
    parser.add_argument('--train-dir', default='./train/', help='Path to training directory')
    parser.add_argument('--load-saved', default=True, type=bool, help='Flag to load')
    parser.add_argument('--gpu-count', default=1, type=int, help='Number of gpu')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of batch')
    parser.add_argument('--pretrained-audio', default=False, type=bool, help='Pretrain U-net model')
    parser.add_argument('--dev-loss-freq', default=1, type=int, help='Number of epochs to print dev loss')
    parser.add_argument('--batch-loss-freq', default=0, type=int, help='If 0 - never prints loss on batches')
    parser.add_argument('--depth', default=7, type=int, help='Depth of model')
    parser.add_argument('--example-type', default='l1', help='What to print on example pictures')
    parser.add_argument('--optimizer', default='SGD', help='Type of optimizer to use')
    parser.add_argument('--audio-activation', default=None, help='Activation for audio model')
    args = parser.parse_args()
    return args


def train(args):
    n_epoch = args.epoch
    n_video = args.N  # count of mixed videos
    n_channels = args.K  # count of video features per pixel
    width = args.width  # width
    height = args.height  # height
    n_frames = int(args.fps * args.fragment_len)  # count of frames in one video
    batch_size = args.batch_size
    audio_pretrained = args.pretrained_audio
    n_gpu = args.gpu_count
    depth = args.depth
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    data_loader = DataLoader(Dataset(height=height, width=width,
            fps=args.fps, frequency=args.freq, fragment_len=args.fragment_len, batch_size=batch_size,
            window_len=args.window_len, overlap_len=args.overlap_len,
            audio_dir=os.path.join(args.train_set_dir, 'audios/train'),
            video_dir=os.path.join(args.train_set_dir, 'videos/train'), random_crop=True, random_shuffle=True),
            batch_size=batch_size)

    data_test_loader = DataLoader(Dataset(height=height, width=width,
            fps=args.fps, frequency=args.freq, fragment_len=args.fragment_len, batch_size=batch_size,
            window_len=args.window_len, overlap_len=args.overlap_len,
            audio_dir=os.path.join(args.dev_set_dir, 'audios/dev'),
            video_dir=os.path.join(args.dev_set_dir, 'videos/dev'), random_crop=False, random_shuffle=False),
            batch_size=batch_size)

    data_eval_loader = DataLoader(Dataset(height=height, width=width,
            fps=args.fps, frequency=args.freq, fragment_len=args.fragment_len,
            batch_size=batch_size,
            window_len=args.window_len, overlap_len=args.overlap_len,
            audio_dir=os.path.join(args.test_set_dir, 'audios/test'),
            video_dir=os.path.join(args.test_set_dir, 'videos/test'), random_crop=False, random_shuffle=False),
            batch_size=batch_size)

    video_model_lr = args.video_model_lr  # learning rate for video model
    audio_model_lr = args.audio_model_lr  # learning rate for audio model
    generator_lr = args.generator_lr  # learning rate for audio generator

    criterion = nn.BCELoss().to(device)

    epoch_loss_freq = args.epoch_loss_freq  # print loss every epoch_loss_freq batches.
    batch_loss_freq = args.batch_loss_freq

    save_freq = args.save_model_freq

    example_freq = args.example_freq
    example_type = args.example_type

    batch_count = len(data_loader)

    path_u = os.path.join(args.train_dir, 'checkpoint/U_model_{}.pt')
    path_v = os.path.join(args.train_dir, 'checkpoint/V_model_{}.pt')
    path_g = os.path.join(args.train_dir, 'checkpoint/G_model_{}.pt')
    path_epoch = os.path.join(args.train_dir, 'checkpoint/epoch.pt')
    os.makedirs(os.path.join(args.train_dir, 'checkpoint/'), exist_ok=True)

    os.makedirs(os.path.join(args.train_dir, "example_images/"), exist_ok=True)

    unet_activation = lambda x: x
    if args.audio_activation.lower() == 'tanh':
        unet_activation = nn.Tanh()
    elif args.audio_activation.lower() == 'sigmoid':
        unet_activation = nn.Sigmoid()
    elif args.audio_activation.lower() == 'no':
        pass
    else:
        raise TypeError("unknown activation")

    start_epoch = 0

    Unet(feature_channels=n_channels, depth=depth).to(device)

    if args.optimizer.lower() == 'adam':
        opt_cls = optim.Adam
        opt_kwargs = {}
    elif args.optimizer.lower() == 'adabound':
        opt_cls = adabound.AdaBound
        opt_kwargs = {}
    elif args.optimizer.lower() == 'sgd':
        opt_cls = optim.SGD
        opt_kwargs = {'momentum': 0.9, 'weight_decay': 1e-4}
    else:
        raise TypeError("unknown optimizer")

    if args.load_saved:
        try:
            start_epoch = torch.load(path_epoch)
            u_model = torch.load(path_u.format(start_epoch)).to(device)
            v_model = torch.load(path_v.format(start_epoch)).to(device)
            g_model = torch.load(path_g.format(start_epoch)).to(device)
            opt_v = opt_cls(v_model.parameters(), lr=video_model_lr, **opt_kwargs)
            opt_u = opt_cls(u_model.parameters(), lr=audio_model_lr, **opt_kwargs)
            opt_g = opt_cls(g_model.parameters(), lr=generator_lr, **opt_kwargs)

        except Exception as e:
            print('Loading failed')
            v_model = Video(n_channels, height, width, n_frames).to(device)
            u_model = Unet(feature_channels=n_channels, depth=depth, audio_activation=unet_activation).to(device)
            g_model = Generator(n_channels).to(device)

            opt_v = opt_cls(v_model.parameters(), lr=video_model_lr, **opt_kwargs)
            opt_u = opt_cls(u_model.parameters(), lr=audio_model_lr, **opt_kwargs)
            opt_g = opt_cls(g_model.parameters(), lr=generator_lr, **opt_kwargs)

            start_epoch = 0
    else:
        v_model = Video(n_channels, height, width, n_frames).to(device)
        u_model = Unet(feature_channels=n_channels, depth=depth, audio_activation=unet_activation).to(device)
        g_model = Generator(n_channels).to(device)

        opt_v = opt_cls(v_model.parameters(), lr=video_model_lr, **opt_kwargs)
        opt_u = opt_cls(u_model.parameters(), lr=audio_model_lr, **opt_kwargs)
        opt_g = opt_cls(g_model.parameters(), lr=generator_lr, **opt_kwargs)

        start_epoch = 0

    if device.type == 'cuda' and n_gpu > 1:
        u_model = nn.DataParallel(u_model, list(range(n_gpu)))
        v_model = nn.DataParallel(v_model, list(range(n_gpu)))
        g_model = nn.DataParallel(g_model, list(range(n_gpu)))

    loss_train = []
    loss_test = []


    test_loss = []
    with torch.no_grad():
        for test_batch_n, test_data in enumerate(data_test_loader, 0):
            audio_sum = test_data[2].to(device) + 1e-10
            for i in range(n_video):
                video = test_data[0][:, i].to(device)

                u_res = u_model(torch.log(audio_sum).detatch())

                video = video.permute([0, 1, 4, 2, 3])
                v_res = v_model(video.detatch())
                g_res = g_model(v_res, u_res)

                weight = torch.log1p(audio_sum).squeeze(1)
                weight = torch.clamp(weight, 1e-3, 10)

                loss = F.binary_cross_entropy((g_res).squeeze(1), (test_data[1][:, i, :].squeeze(1) > (audio_sum.squeeze(1) / N)).type(torch.Tensor).to(device), weight.to(device)).to(device)

                test_loss.append(loss.data.item())

        print('epoch [{} / {}]\t Test loss: {}'.format(-1, n_epoch, np.array(test_loss).mean()))

    os.makedirs(os.path.join(args.train_dir, 'logs/'), exist_ok=True)
    weight_log_file = open(os.path.join(args.train_dir, 'logs/weights.txt'), 'w')
    res_log_file = open(os.path.join(args.train_dir, 'logs/res.txt'), 'w')

    start_time = time.time()
    for epoch in range(start_epoch, n_epoch):
        print('\nepoch: {}\n'.format(epoch + 1))
        for batch_n, data in enumerate(data_loader, 0):
            audio_sum = data[2].to(device) + 1e-10

            losses = []
            for i in range(n_video):
                u_model.zero_grad()
                v_model.zero_grad()
                g_model.zero_grad()

                video = data[0][:, i].to(device)

                u_res = u_model(torch.log(audio_sum).detatch())

                video = video.permute([0, 1, 4, 2, 3])

                v_res = v_model(video)
                g_res = g_model(v_res, u_res)
                print('U_RES:', u_res.shape, u_res.mean().item(), u_res.min().item(), u_res.max().item(), u_res.std().item(), file=res_log_file)
                print('V_RES:', v_res.shape, v_res.mean().item(), v_res.min().item(), v_res.max().item(),
                      v_res.std().item(), file=res_log_file)
                print('G_RES:', g_res.shape, g_res.mean().item(), g_res.min().item(), g_res.max().item(),
                      g_res.std().item(), file=res_log_file)

                weight = torch.log1p(audio_sum).squeeze(1)
                weight = torch.clamp(weight, 1e-3, 10)

                loss = F.binary_cross_entropy(g_res.squeeze(1), (data[1][:, i, :].squeeze(1) > (audio_sum.squeeze(1) / N)).type(torch.Tensor).to(device), weight=weight.to(device)).to(device)
                losses.append(loss.data.item())
                loss.backward()

                opt_v.step()
                opt_u.step()
                opt_g.step()

            loss_train.append(np.array(losses).mean())

            if batch_loss_freq != 0 and (batch_n + 1) % batch_loss_freq == 0:
                print('batch: {:<10}   |   Loss: {:<20}    |    average time per batch: {:<20}'.format(batch_n + 1, np.array(losses).mean(), (time.time() - start_time) / (batch_n + 1)))

        if (epoch + 1) % epoch_loss_freq == 0:
            print('epoch: [{:<4} / {:<4}]   |   TRAIN_LOSS: {:<20}   |   average time per epoch: {}'.format(epoch + 1, n_epoch, np.array(losses).mean(), (time.time() - start_time) / (epoch + 1)))
            print('WEIGHTS:', g_model.weights, file=weight_log_file)
            print('BIAS:', g_model.bias, file=weight_log_file)
            print('WEIGHTS GRAD:', g_model.weights.grad, file=weight_log_file)
            print('BIAS GRAD:', g_model.bias.grad, file=weight_log_file)

        if (epoch + 1) % args.dev_loss_freq == 0:
            test_loss = []
            with torch.no_grad():
                for test_batch_n, test_data in enumerate(data_test_loader, 0):
                    audio_sum = test_data[2].to(device) + 1e-10
                    for i in range(n_video):
                        video = test_data[0][:, i].to(device)

                        u_res = u_model(torch.log(audio_sum).detatch())

                        video = video.permute([0, 1, 4, 2, 3])
                        v_res = v_model(video)
                        g_res = g_model(v_res, u_res)

                        weight = torch.log1p(audio_sum).squeeze(1)
                        weight = torch.clamp(weight, 1e-3, 10)

                        loss = F.binary_cross_entropy(g_res.squeeze(1), (test_data[1][:, i, :].squeeze(1) > (audio_sum.squeeze(1) / N)).type(torch.Tensor).to(device), weight.to(device)).to(device)

                        test_loss.append(loss.data.item())

            loss_test.append(np.array(test_loss).mean())
            print('epoch [{} / {}]\t Test loss: {}'.format(epoch + 1, n_epoch, np.array(test_loss).mean()))

        if (epoch + 1) % example_freq == 0:
            with torch.no_grad():
                for eval_batch_n, eval_data in enumerate(data_eval_loader, 0):
                    if eval_batch_n == 0:
                        picture = eval_data[0][-1, 0, -1, :, :, :].to(device)
                        video = eval_data[0][-1:, 0, :, :, :, :].to(device)
                        audio = eval_data[1][-1:, 0, :].to(device)

                        picture = picture.permute([2, 0, 1])
                        video = video.permute([0, 1, 4, 2, 3])

                        u_sample_res = u_model(torch.log(audio_sum).detatch())
                        v_sample_res = v_model(video)
                        g_sample_res = g_model.forward_pixelwise(v_sample_res, u_sample_res)

                        model_sample_answer = torch.mul(g_sample_res, audio_sum[:, None, :, :, :])

                        located_sound_picture = get_picture_from_model_ans(model_sample_answer, example_type,
                                                                           picture.shape, device)

                        output = located_sound_picture * 0.3 + picture * 0.7

                        located_sound_picture = located_sound_picture.squeeze()
                        picture = picture.squeeze()
                        output = output.squeeze()
                        plt.imsave(os.path.join(args.train_dir, "example_images/epoch_sound_{}.png".format(epoch)),
                                   np.transpose(located_sound_picture.cpu().numpy(), (1, 2, 0)))
                        plt.imsave(os.path.join(args.train_dir, "example_images/epoch_source_{}.png".format(epoch)),
                                   np.transpose(picture.cpu().numpy(), (1, 2, 0)))

                        plt.imsave(os.path.join(args.train_dir, "example_images/epoch_final_{}.png".format(epoch)),
                                   np.transpose(output.cpu().numpy(), (1, 2, 0)))
                        print("Example saved")
                    else:
                        break

        if (epoch + 1) % save_freq == 0:
            print('Saving model')
            torch.save(u_model, path_u.format(epoch))
            torch.save(v_model, path_v.format(epoch))
            torch.save(g_model, path_g.format(epoch))
            torch.save(epoch, path_epoch)
            print('Model saved')


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
