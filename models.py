import torchvision.models as models
import torch.nn as nn
import torch

class Video(nn.Module):
    def __init__(self, n_channels, height, width, n_frames):
        super(Video, self).__init__()

        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.n_frames = n_frames

        self.main = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

        self.main[-1][0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.main[-1][0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.main[-1][1].conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), padding=(2, 2), bias=False)
        self.main[-1][1].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), padding=(2, 2), bias=False)

        def init(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.main.add_module("conv_k", nn.Conv2d(512, n_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.tmp_conv = nn.Conv3d(n_frames, 1, kernel_size=(1, 1, 1))

        self.activation = nn.Sigmoid()

        self.main[-2].apply(init)
        self.main[-1].apply(init)
        self.tmp_conv.apply(init)

    def forward(self, input):
        x = input.reshape((-1, 3, self.height, self.width))
        x = self.main(x)

        x = x.reshape((-1, self.n_frames, self.n_channels, self.height // 16, self.width // 16))

        x = self.tmp_conv(x).squeeze()
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_channels):
        super(Generator, self).__init__()

        self.n_channels = n_channels

        self.main = nn.Linear(n_channels, 1)

        self.activation = nn.Sigmoid()

    def forward(self, inputV, inputA):  # inputV.shape = [bs, K, h // 16, w // 16], inputA.shape = [bs, K, audH, audW]
        x = inputV[:, :, :, :, None, None] * inputA[:, :, None, None, :, :]
        x = x.permute((0, 2, 3, 4, 5, 1))
        x = self.main(x).squeeze()

        x = self.activation(x)
        return x  # x.shape = [bs, h // 16, w // 16, audH, audW]


