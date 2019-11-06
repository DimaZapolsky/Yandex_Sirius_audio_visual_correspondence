import models

import numpy as np
import sklearn.decomposition

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

NUM_EPOCHS = 10
N = 2 #count of mixed videos
K = 16 #count of video features per pixel
W = 224 #width
H = 224 #height
T = 6 #count of frames in one video

lr1 = 1e-4 #learning rate for video model
lr2 = 1e-3 #learning rate for audio model
lr3 = 1e-3 #learning rate for audio generator

V = Video(K, H, W, T)
U = Unet()
G = Generator(K, H, W, Aud_H, Aud_W)

opt_V = optim.SGD(V.parameters(), lr=lr1)
opt_U = optim.SGD(U.parameters(), lr=lr2)
opt_G = optim.SGD(G.parameters(), lr=lr3)

criterion = nn.L1Loss()

print_loss_freq = 50 #print loss every print_loss_freq batches.

save_freq = 300

example_freq = 400

batch_count = len(dataloader)

PATH_U = "~/training/checkpoints/U_model_epoch"
PATH_V = "~/training/checkpoints/V_model_epoch"
PATH_G = "~/training/checkpoints/G_model_epoch"
PATH_EPOCH = "~/training/checkpoints/epoch_number"
start_epoch = 0
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

from_save = 0

if (from_save == 1):
    U = torch.load(PATH_U)
    V = torch.load(PATH_V)
    G = torch.load(PATH_G)
    start_epoch = torch.load(PATH_EPOCH)

if (device.type == 'cuda') and (ngpu > 1):
    U = nn.DataParallel(U, list(range(ngpu)))
    V = nn.DataParallel(V, list(range(ngpu)))
    G = nn.DataParallel(G, list(range(ngpu)))


loss_train = []
loss_test = []

for epoch in range(start_epoch, NUM_EPOCHS):
    for batch_n, data in enumerate(dataloader, 0):
        audio_sum = torch.sum(data[1], 1).to(device)
        losses = []
        for i in range(N):
            U.zero_grad()
            V.zero_grad()
            G.zero_grad()

            video = data[0][:, i].to(device)

            U_res = U(audio_sum)
            V_res = V(video)
            G_res = G(V_res, U_res)  # (bs, x, y, t, freq)
            model_answer = torch.mul(G_res, audio_sum[None, 1, 1, None, None])  # (bs, x, y, t, freq) * (bs, t, freq)

            loss = criterion(torch.sum(model_answer, [1, 2]), data[1][:, i, :] / audio_sum)
            losses.append(loss)
            loss.backward()

            opt_V.step()
            opt_U.step()
            opt_G.step()
        loss_train.append(np.array(losses).mean())

        if (batch_n % print_loss_freq == 0):
            test_loss = []
            with torch.no_grad():
                for batch_n, data in enumerate(data_test_loader, 0):
                    audio_sum = torch.sum(data[1], 1).to(device)
                    for i in range(N):
                        video = data[0][:, i].to(device)

                        U_res = U(audio_sum)
                        V_res = V(video)
                        G_res = G(V_res, U_res)#(bs, x, y, t, freq)
                        model_answer = torch.mul(G_res, audio_sum[None, 1, 1, None, None])#(bs, x, y, t, freq) * (bs, t, freq)

                        loss = criterion(torch.sum(model_answer, [1, 2]), data[1][:, i, :] / audio_sum)
                        test_loss.append(loss)
            loss_test.append(np.array(test_loss).mean())
            print('epoch [%d/%d]\t batch [%d/%d]\t. Train loss: %d,\t test loss: %d' % (epoch, NUM_EPOCHS, batch_n, batch_count, np.array(losses).mean()), np.array(test_loss).mean())

        if (batch_n % save_freq == 0):
            torch.save(U, PATH_U)
            torch.save(V, PATH_V)
            torch.save(G, PATH_G)
            torch.save(epoch, PATH_EPOCH)


        if (batch_n % example_freq == 0):
            with torch.no_grad():
                picture = data[0][-1, 0, :, :, :, -1]
                video = data[0][-1:, 0, :, :, :, :]
                audio = data[1][-1:, 0, :]

                U_sample_res = U(audio)
                V_sample_res = V(video)
                G_sample_res = G(V_sample_res, U_sample_res)

                model_sample_answer = torch.mul(G_sample_res, audio[None, 1, 1, None, None])

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
                output = (located_sound_picture * 0.2 + picture * 0.8)

                fig = plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.imshow(np.transpose(output.numpy(), (1, 2, 0)))

