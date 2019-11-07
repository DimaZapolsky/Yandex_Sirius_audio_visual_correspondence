import models
import numpy as np
import sklearn.decomposition
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', required=True, type=int, help='Video\'s fps')
    parser.add_argument('--width', required=True, type=int, help='Image width')
    parser.add_argument('--height', required=True, type=int, help='Image height')
    parser.add_argument('--freq', required=True, type=int, help='Audio frequency')
    parser.add_argument('--train-dir', default='./', help='Train dataset directory')
    parser.add_argument('--dev-dir', default='./', help='Dev dataset directory')
    parser.add_argument('--test-dir', default='./', help='Test dataset directory')
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
    parser.add_argument('--train-dir', default='./', help='Path to training directory')
    parser.add_argument('--load-saved', default=True, type=bool, help='Flag to load')
    parser.add_argument('--gpu-count', default=1, type=int, help='Number of gpu')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of batch')
    parser.add_argument('--pretrained', default=False, type=bool, help='Pretrain U-net model')
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
    audio_pretrained = args.pretrained

    data_loader = DataLoader(Dataset(height, width,
            args.fps, args.freq, args.fragment_len, batch_size,
            args.window_len, args.overlap_len, audio_dir=os.path.join(args.train_dir, 'audio/train'),
            video_dir=os.path.join(args.train_dir, 'video/train')), batch_size=batch_size)

    data_test_loader = DataLoader(Dataset(height, width,
            args.fps, args.freq, args.fragment_len, batch_size,
            args.window_len, args.overlap_len, audio_dir=os.path.join(args.test_dir, 'audio/test'),
            video_dir=os.path.join(args.test_dir, 'video/test')), batch_size=batch_size)

    video_model_lr = args.video_model_lr  # learning rate for video model
    audio_model_lr = args.audio_model_lr  # learning rate for audio model
    generator_lr = args.generator_lr  # learning rate for audio generator

    v_model = Video(K, H, width, n_frames)
    u_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=n_channels, init_features=32, pretrained=audio_pretrained)
    g_model = Generator(K)

    opt_v = optim.SGD(v_model.parameters(), lr=video_model_lr)
    opt_u = optim.SGD(u_model.parameters(), lr=audio_model_lr)
    opt_g = optim.SGD(g_model.parameters(), lr=generator_lr)

    criterion = nn.L1Loss()

    print_loss_freq = args.print_loss_freq  # print loss every print_loss_freq batches.

    save_freq = args.save_model_freq

    example_freq = args.example_freq

    batch_count = len(data_loader)

    path_u = os.path.join(args.train_dir, 'checkpoint/U_model_{}.pt')
    os.makedirs(path_u)
    path_v = os.path.join(args.train_dir, 'checkpoint/V_model_{}.pt')
    os.makedirs(path_v)
    path_g = os.path.join(args.train_dir, 'checkpoint/G_model_{}.pt')
    os.makedirs(path_g)
    path_epoch = os.path.join(args.train_dir, 'checkpoint/epoch.pt')
    start_epoch = 0
    n_gpu = args.gpu_count
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    if args.load_saved:
        start_epoch = torch.load(path_epoch)
        u_model = torch.load(path_u.format(start_epoch))
        v_model = torch.load(path_v.format(start_epoch))
        g_model = torch.load(path_g.format(start_epoch))

    if device.type == 'cuda' and n_gpu > 1:
        u_model = nn.DataParallel(u_model, list(range(n_gpu)))
        v_model = nn.DataParallel(v_model, list(range(n_gpu)))
        g_model = nn.DataParallel(g_model, list(range(n_gpu)))

    loss_train = []
    loss_test = []

    for epoch in range(start_epoch, n_epoch):
        for batch_n, data in enumerate(data_loader, 0):
            audio_sum = data[2].to(device)
            losses = []
            for i in range(n_video):
                u_model.zero_grad()
                v_model.zero_grad()
                g_model.zero_grad()

                video = data[0][:, i].to(device)

                u_res = u_model(audio_sum)
                v_res = v_model(video)
                g_res = g_model(v_res, u_res)  # (bs, x, y, t, freq)
                model_answer = torch.mul(g_res, audio_sum[None, 1, 1, None, None])  # (bs, x, y, t, freq) * (bs, t, freq)

                loss = criterion(torch.sum(model_answer, [1, 2]), data[1][:, i, :] / audio_sum)
                losses.append(loss)
                loss.backward()

                opt_v.step()
                opt_u.step()
                opt_g.step()

            loss_train.append(np.array(losses).mean())

            if batch_n % print_loss_freq == 0:
                test_loss = []
                with torch.no_grad():
                    for test_batch_n, test_data in enumerate(data_test_loader, 0):
                        audio_sum = torch.sum(test_data[1], 1).to(device)
                        for i in range(n_video):
                            video = test_data[0][:, i].to(device)

                            u_res = u_model(audio_sum)
                            v_res = v_model(video)
                            g_res = g_model(v_res, u_res)  # (bs, x, y, t, freq)
                            model_answer = torch.mul(g_res, audio_sum[None, 1, 1, None, None])  # (bs, x, y, t, freq) * (bs, t, freq)

                            loss = criterion(torch.sum(model_answer, [1, 2]), test_data[1][:, i, :] / audio_sum)
                            test_loss.append(loss)
                loss_test.append(np.array(test_loss).mean())
                print('epoch [%d/%d]\t batch [%d/%d]\t. Train loss: %d,\t test loss: %d' % (epoch, n_epoch, batch_n, batch_count, np.array(losses).mean(), np.array(test_loss).mean())

            if batch_n % save_freq == 0:
                torch.save(u_model, path_u)
                torch.save(v_model, path_v)
                torch.save(g_model, path_g)
                torch.save(epoch, path_epoch)


            if batch_n % example_freq == 0:
                with torch.no_grad():
                    picture = data[0][-1, 0, :, :, :, -1]
                    video = data[0][-1:, 0, :, :, :, :]
                    audio = data[1][-1:, 0, :]

                    u_sample_res = u_model(audio)
                    v_sample_res = v_model(video)
                    g_sample_res = g_model(v_sample_res, u_sample_res)

                    model_sample_answer = torch.mul(g_sample_res, audio[None, 1, 1, None, None])

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
                    plt.imshow(np.transpose(output.numpy(), (1, 2, 0)))


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__';
    main()
