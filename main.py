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
    parser.add_argument('--print-loss-freq', default=50, type=int, help='Number of batches to print losses')
    parser.add_argument('--save-model-freq', default=300, type=int, help='Number of batches to save model')
    parser.add_argument('--example-freq', default=400, type=int, help='Batches to print example')
    parser.add_argument('--train-dir', default='./train/', help='Path to training directory')
    parser.add_argument('--load-saved', default=True, type=bool, help='Flag to load')
    parser.add_argument('--gpu-count', default=1, type=int, help='Number of gpu')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of batch')
    parser.add_argument('--pretrained-audio', default=False, type=bool, help='Pretrain U-net model')
    parser.add_argument('--dev-loss-freq', default=1, type=int, help='Number of epochs to print dev loss')
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
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    data_loader = DataLoader(Dataset(height=height, width=width,
            fps=args.fps, frequency=args.freq, fragment_len=args.fragment_len, batch_size=batch_size,
            window_len=args.window_len, overlap_len=args.overlap_len,
            audio_dir=os.path.join(args.train_set_dir, 'audios/train'),
            video_dir=os.path.join(args.train_set_dir, 'videos/train')), batch_size=batch_size)

    data_test_loader = DataLoader(Dataset(height=height, width=width,
            fps=args.fps, frequency=args.freq, fragment_len=args.fragment_len, batch_size=batch_size,
            window_len=args.window_len, overlap_len=args.overlap_len,
            audio_dir=os.path.join(args.dev_set_dir, 'audios/dev'),
            video_dir=os.path.join(args.dev_set_dir, 'videos/dev')), batch_size=batch_size)

    video_model_lr = args.video_model_lr  # learning rate for video model
    audio_model_lr = args.audio_model_lr  # learning rate for audio model
    generator_lr = args.generator_lr  # learning rate for audio generator

    criterion = nn.BCELoss().to(device)

    print_loss_freq = args.print_loss_freq  # print loss every print_loss_freq batches.

    save_freq = args.save_model_freq

    example_freq = args.example_freq

    batch_count = len(data_loader)

    path_u = os.path.join(args.train_dir, 'checkpoint/U_model_{}.pt')
    path_v = os.path.join(args.train_dir, 'checkpoint/V_model_{}.pt')
    path_g = os.path.join(args.train_dir, 'checkpoint/G_model_{}.pt')
    path_epoch = os.path.join(args.train_dir, 'checkpoint/epoch.pt')
    os.makedirs(os.path.join(args.train_dir, 'checkpoint/'), exist_ok=True)

    start_epoch = 0

    Audio(n_channels).to(device)

    if args.load_saved:
        try:
            start_epoch = torch.load(path_epoch)
            u_model = torch.load(path_u.format(start_epoch)).to(device)
            v_model = torch.load(path_v.format(start_epoch)).to(device)
            g_model = torch.load(path_g.format(start_epoch)).to(device)
            opt_v = adabound.AdaBound(v_model.parameters(), lr=video_model_lr)
            opt_u = adabound.AdaBound(u_model.parameters(), lr=audio_model_lr)
            opt_g = adabound.AdaBound(g_model.parameters(), lr=generator_lr)

        except Exception as e:
            print('Loading failed')
            v_model = Video(n_channels, height, width, n_frames).to(device)
            u_model = Audio(n_channels).to(device)
            g_model = Generator(n_channels).to(device)

            opt_v = adabound.AdaBound(v_model.parameters(), lr=video_model_lr)
            opt_u = adabound.AdaBound(u_model.parameters(), lr=audio_model_lr)
            opt_g = adabound.AdaBound(g_model.parameters(), lr=generator_lr)

            start_epoch = 0
    else:
        v_model = Video(n_channels, height, width, n_frames).to(device)
        u_model = Audio(n_channels).to(device)
        g_model = Generator(n_channels).to(device)

        opt_v = adabound.AdaBound(v_model.parameters(), lr=video_model_lr)
        opt_u = adabound.AdaBound(u_model.parameters(), lr=audio_model_lr)
        opt_g = adabound.AdaBound(g_model.parameters(), lr=generator_lr)

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

                u_res = u_model(audio_sum)

                video = video.permute([0, 1, 4, 2, 3])
                v_res = v_model(video)
                g_res = g_model(v_res, u_res)
                #print(g_res.shape, (data[1][:, i, :].squeeze(1) > data[1][:, 1 - i, :].squeeze(1)).type(torch.Tensor).shape) 

                weight = torch.log1p(audio_sum).squeeze(1)
                weight = torch.clamp(weight, 1e-3, 10)

                loss = F.binary_cross_entropy((g_res).squeeze(1), (test_data[1][:, i, :].squeeze(1) > test_data[1][:, 1 - i, :].squeeze(1)).type(torch.Tensor).to(device), weight.to(device)).to(device)

                test_loss.append(loss.data.item())
 
        print('epoch [{} / {}]\t Test loss: {}'.format(-1, n_epoch, np.array(test_loss).mean()))


    for epoch in range(start_epoch, n_epoch):
        start_time = time.time()
        print('\nepoch: {}\n'.format(epoch))
        for batch_n, data in enumerate(data_loader, 0):
            audio_sum = data[2].to(device) + 1e-10

            losses = []
            for i in range(n_video):
                u_model.zero_grad()
                v_model.zero_grad()
                g_model.zero_grad()

                video = data[0][:, i].to(device)

                # print('tut')
                # print(audio_sum.mean())
                # print(audio_sum.min())
                # print(audio_sum.max())
                #
                # plt.imsave(
                #     fname='spectrogram.png',
                #     arr=audio_sum.squeeze() / audio_sum.max(),
                # )
                u_res = u_model(audio_sum)

                video = video.permute([0, 1, 4, 2, 3])
                v_res = v_model(video)
                g_res = g_model(v_res, u_res)

                weight = torch.log1p(audio_sum).squeeze(1)
                weight = torch.clamp(weight, 1e-3, 10)

                loss = F.binary_cross_entropy((g_res).squeeze(1), (data[1][:, i, :].squeeze(1) > data[1][:, 1 - i, :].squeeze(1)).type(torch.Tensor).to(device), weight=weight.to(device)).to(device)
                losses.append(loss.data.item())
                loss.backward()

                opt_v.step()
                opt_u.step()
                opt_g.step()

            if (batch_n + 1) % print_loss_freq == 0:
                print('batch: {:<10}   |   loss: {:<20}    |    average time per batch: {:<20}'.format(batch_n + 1, np.array(losses).mean(), (time.time() - start_time) / (batch_n + 1)))
                with open('grad_log.txt', 'a') as file:
                    print(g_model.weights, file=file)
                    print(g_model.bias, file=file)
                    print(g_model.tuner, file=file)
                # print(g_model.weights, file=open('log.txt', 'w'))

            loss_train.append(np.array(losses).mean())

        if (epoch + 1) % args.dev_loss_freq == 0:
            test_loss = []
            with torch.no_grad():
                for test_batch_n, test_data in enumerate(data_test_loader, 0):
                    audio_sum = test_data[2].to(device) + 1e-10
                    for i in range(n_video):
                        video = test_data[0][:, i].to(device)

                        u_res = u_model(audio_sum)

                        video = video.permute([0, 1, 4, 2, 3])
                        v_res = v_model(video)
                        g_res = g_model(v_res, u_res)
                        #print(g_res.shape, (data[1][:, i, :].squeeze(1) > data[1][:, 1 - i, :].squeeze(1)).type(torch.Tensor).shape) 

                        weight = torch.log1p(audio_sum).squeeze(1)
                        weight = torch.clamp(weight, 1e-3, 10)

                        loss = F.binary_cross_entropy((g_res).squeeze(1), (test_data[1][:, i, :].squeeze(1) > test_data[1][:, 1 - i, :].squeeze(1)).type(torch.Tensor).to(device), weight.to(device)).to(device)

                        test_loss.append(loss.data.item())

                    with torch.no_grad():
                        if (test_batch_n != 0):
                            continue
                        picture = data[0][-1, 0, :, :, :, -1]
                        video = data[0][-1:, 0, :, :, :, :]
                        audio = data[1][-1:, 0, :]

                        u_sample_res = u_model(audio_sum)
                        v_sample_res = v_model(video)
                        g_sample_res = g_model.forward_pixelwise(v_sample_res, u_sample_res)

                        model_sample_answer = torch.mul(g_sample_res, audio_sum[:, None, None, :, :])

                        pca = sklearn.decomposition.PCA(n_components=3)
                        vectors_square = model_sample_answer[-1, :, :, -1, :]
                        vectors_flatten = vectors_square.reshape(-1, vectors_square.shape()[-1]).numpy()
                        rgb = pca.fit_transform(vectors_flatten)

                        rgb_picture = np.reshape(rgb, vecrors_square.shape(0)[:2] + (-1,))
                        located_sound_picture = np.transpose(rgb_picture, [2, 0, 1])
                        full_sound = torch.from_numpy(located_sound_picture)
                        full_sound = full_sound[None, :, :, :]

                        x_out = torch.zeros((1,) + picture.shape[1:] + (2,))
                        nx = np.linspace(-1, 1, picture.shape[1])
                        ny = np.linspace(-1, 1, picture.shape[2])
                        nxv, nyv = np.meshgrid(nx, ny)
                        x_out[:, :, :, 0] = torch.from_numpy(nxv)
                        x_out[:, :, :, 1] = torch.from_numpy(nyv)

                        located_sound_picture = F.grid_sample(full_sound, x_out)
                        output = located_sound_picture * 0.3 + picture * 0.7

                        fig = plt.figure(figsize=(8,8))
                        plt.axis("off")
                        plt.imsave("example_images/epoch_{}.png".format(epoch), np.transpose(output.numpy(), (1, 2, 0)))
                        print("Example saved")


            loss_test.append(np.array(test_loss).mean())
            print('epoch [{} / {}]\t Test loss: {}'.format(epoch, n_epoch, np.array(test_loss).mean()))

            if (batch_n + 1) % example_freq == 0:
                continue
                with torch.no_grad():
                    picture = data[0][-1, 0, :, :, :, -1]
                    video = data[0][-1:, 0, :, :, :, :]
                    audio = data[1][-1:, 0, :]

                    u_sample_res = u_model(audio_sum)
                    v_sample_res = v_model(video)
                    g_sample_res = g_model.forward_pixelwise(v_sample_res, u_sample_res)

                    model_sample_answer = torch.mul(g_sample_res, audio_sum[:, None, None, :, :])

                    pca = sklearn.decomposition.PCA(n_components=3)
                    vectors_square = model_sample_answer[-1, :, :, -1, :]
                    vectors_flatten = vectors_square.reshape(-1, vectors_square.shape()[-1]).numpy()
                    rgb = pca.fit_transform(vectors_flatten)

                    rgb_picture = np.reshape(rgb, vecrors_square.shape(0)[:2] + (-1,))
                    located_sound_picture = np.transpose(rgb_picture, [2, 0, 1])
                    full_sound = torch.from_numpy(located_sound_picture)
                    full_sound = full_sound[None, :, :, :]

                    x_out = torch.zeros((1,) + picture.shape[1:] + (2,))
                    nx = np.linspace(-1, 1, picture.shape[1])
                    ny = np.linspace(-1, 1, picture.shape[2])
                    nxv, nyv = np.meshgrid(nx, ny)
                    x_out[:, :, :, 0] = torch.from_numpy(nxv)
                    x_out[:, :, :, 1] = torch.from_numpy(nyv)

                    located_sound_picture = F.grid_sample(full_sound, x_out)
                    output = located_sound_picture * 0.2 + picture * 0.8

                    fig = plt.figure(figsize=(8,8))
                    plt.axis("off")
                    plt.imsave("example_images/epoch_{}.png".format(epoch), np.transpose(output.numpy(), (1, 2, 0)))

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
