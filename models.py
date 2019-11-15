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

        def init(m):
            if m.__class__.__name__.find('Conv') != -1:
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                    if m.kernel_size == (3, 3):
                        m.dilation = (1, 1)
                        m.padding = (1, 1)
                else:
                    if (m.kernel_size == (3, 3)):
                        m.dilation = (2, 2)
                        m.padding = (2, 2)

        self.main[-1].apply(init)

        self.main.add_module("conv_k", nn.Conv2d(512, n_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.tmp_conv = nn.AdaptiveMaxPool3d(output_size=(1, height // 16, width // 16))
        self.tmp_conv_field = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

        self.activation = nn.Sigmoid()

        #self.main[-2].apply(init)
        #self.main[-1].apply(init)
        #self.tmp_conv.apply(init)

    def forward(self, input):
        x = input.reshape((-1, 3, self.height, self.width))
        x = self.main(x)

        x = x.reshape((-1, self.n_frames, self.n_channels, self.height // 16, self.width // 16))

        x = x.permute((0, 2, 1, 3, 4))
        x = self.tmp_conv_field(x).squeeze(2)
        x = self.activation(x)
        return x

    def forward_eval(self, input):
        x = input.reshape((-1, 3, self.height, self.width))
        x = self.main(x)

        x = x.reshape((-1, self.n_frames, self.n_channels, self.height // 16, self.width // 16))

        x = x.permute((0, 2, 1, 3, 4))
        x = self.tmp_conv(x).squeeze(2)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_channels):
        super(Generator, self).__init__()

        self.n_channels = n_channels

        self.weights = nn.Parameter(torch.Tensor(1, 1, 1, n_channels))
        self.bias = nn.Parameter(torch.Tensor(1))

        self.activation = nn.Sigmoid()

        self.tuner = nn.Parameter(torch.Tensor(1))

        nn.init.constant_(self.weights, 1)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.tuner, 1)

    def forward(self, inputV, inputA):  # inputV.shape = [bs, K, h // 16, w // 16], inputA.shape = [bs, K, audH, audW]
        input_V_flattened = inputV.view(inputV.shape[0], self.n_channels, -1)
        input_V_flattened, _ = input_V_flattened.max(2, keepdim=True)
        input_A_flattened = inputA.view(inputA.shape[0], self.n_channels, -1)

        input_V_flattened = input_V_flattened.permute([0, 2, 1])
        input_V_flattened = input_V_flattened * self.weights
        x = input_V_flattened @ input_A_flattened + self.bias

        x = x.view((-1,) + inputA.shape[-2:])
        #x = torch.mean(x, [1, 2])

        #x = torch.sigmoid(x) #  * self.tuner

        x = self.activation(x)
        return x # x.shape = [bs, audH, audW]

    def forward_pixelwise(self, inputV, inputA):

        input_V_flattened = inputV.view(inputV.shape[0], self.n_channels, -1)

        input_A_flattened = inputA.view(inputA.shape[0], self.n_channels, -1)

        input_V_flattened = input_V_flattened.permute([0, 2, 1])
        input_V_flattened = input_V_flattened * self.weights
        x = input_V_flattened @ input_A_flattened + self.bias

        x = x.view((-1,) + inputV.shape[-2:] + inputA.shape[-2:])
        #x = torch.mean(x, [1, 2])

        #x = torch.sigmoid(x) #  * self.tuner

        x = self.activation(x)
        return x


class Audio(nn.Module):
    def __init__(self, n_channels):
        super(Audio, self).__init__()

        self.main = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=n_channels, init_features=32, pretrained=False)

    def forward(self, input):
        return self.main(input)


