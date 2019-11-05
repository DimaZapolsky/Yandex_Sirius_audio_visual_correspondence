import torchvision.models as models
import torch.nn as nn
import torch

class Video(nn.Module):
    def __init__(self, K, H, W, T):
        super(Video, self).__init__()

        self.K = K
        self.H = H
        self.W = W
        self.T = T

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

        self.main.add_module("conv_k", nn.Conv2d(512, K, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.tmp_conv = nn.Conv3d(K, K, kernel_size=(6, 1, 1))

        self.main[-2].apply(init)
        self.main[-1].apply(init)
        self.tmp_conv.apply(init)

    def forward(self, input):
        x = input.reshape((-1, 3, self.H, self.W))
        x = self.main(x)
        x = x.reshape((-1, self.T, self.K, self.H // 16, self.W // 16))
        x = x.permute((0, 2, 1, 3, 4))
        x = self.tmp_conv(x).squeeze()
        return x

