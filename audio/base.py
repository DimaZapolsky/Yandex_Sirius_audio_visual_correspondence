import torch


class Unet(torch.nn.Module):
    def __init__(
            self,
            feature_channels=64,  # hyperparameter: number of feature channels between encoder and decoder
            depth=5,  # hyperparameter: number of unet convolutions
            generated_features=64,  # hyperparameter: final number of feature maps
            use_dropout=False,
            dropout_p=0.5,
            audio_activation=torch.nn.Tanh,
    ):
        super(Unet, self).__init__()
        self.min_depth = 5
        self.batch_norm = torch.nn.BatchNorm2d(1)

        self.unet_block = UnetBlock(
            output_channels=generated_features * 8,
            inner_input_channels=generated_features * 8,
            innermost=True,
            dropout_p=dropout_p,
        )

        for _ in range(depth - self.min_depth):
            self.unet_block = UnetBlock(
                output_channels=generated_features * 8,
                inner_input_channels=generated_features * 8,
                submodule=self.unet_block,
                use_dropout=use_dropout,
                dropout_p=dropout_p,
            )

        self.unet_block = UnetBlock(
            output_channels=generated_features * 4,
            inner_input_channels=generated_features * 8,
            submodule=self.unet_block,
            dropout_p=dropout_p,
        )
        self.unet_block = UnetBlock(
            output_channels=generated_features * 2,
            inner_input_channels=generated_features * 4,
            submodule=self.unet_block,
            dropout_p=dropout_p,
        )
        self.unet_block = UnetBlock(
            output_channels=generated_features,
            inner_input_channels=generated_features * 2,
            submodule=self.unet_block,
            dropout_p=dropout_p,
        )
        self.unet_block = UnetBlock(
            output_channels=feature_channels,
            inner_input_channels=generated_features,
            input_channels=1,
            submodule=self.unet_block,
            outermost=True,
            dropout_p=dropout_p,
        )
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data.normal_(0.0, 0.001)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        self.unet_block.apply(init_weights)
        self.batch_norm.apply(init_weights)
        self.activation = audio_activation

    def forward(self, input_data):
        output_data = self.batch_norm(input_data)
        output_data = self.unet_block(output_data)
        return self.activation(output_data)


class UnetBlock(torch.nn.Module):
    def __init__(
            self,
            output_channels, inner_input_channels,
            input_channels=None, inner_output_channels=None,
            outermost=False, innermost=False,
            use_dropout=False, noskip=False,
            submodule=None,
            dropout_p=0.5,
    ):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False

        if input_channels is None:
            input_channels = output_channels
        if innermost:
            inner_output_channels = inner_input_channels
        elif inner_output_channels is None:
            inner_output_channels = 2 * inner_input_channels

        down_relu = torch.nn.LeakyReLU(0.2, True)
        down_norm = torch.nn.BatchNorm2d(inner_input_channels)

        up_relu = torch.nn.ReLU(True)
        up_norm = torch.nn.BatchNorm2d(output_channels)
        up_sample = torch.nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

        down_conv = torch.nn.Conv2d(
            input_channels,
            inner_input_channels,
            kernel_size=4,
            stride=2, padding=1,
            bias=use_bias
        )
        down_conv2 = torch.nn.Conv2d(
            inner_input_channels,
            inner_input_channels,
            kernel_size=3,
            padding=1,
        )
        if outermost:
            up_conv = torch.nn.Conv2d(
                inner_output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias
            )
            up_conv2 = torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1
            )
            down = [down_conv, down_relu, down_conv2]
            up = [up_relu, up_sample, up_conv, up_relu, up_conv2]

        elif innermost:
            up_conv = torch.nn.Conv2d(
                inner_output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias
            )
            up_conv2 = torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias
            )
            down = [down_relu, down_conv, down_relu, down_conv2]
            up = [up_relu, up_sample, up_conv, up_norm, up_relu, up_conv2]

        else:
            up_conv = torch.nn.Conv2d(
                inner_output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias
            )
            up_conv2 = torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias
            )
            down = [down_relu, down_conv, down_norm, down_relu, down_conv2]
            up = [up_relu, up_sample, up_conv, up_norm, up_relu, up_conv2]

        model = down
        if submodule:
            model += [submodule]
        model += up
        if use_dropout:
            model += [torch.nn.Dropout(dropout_p)]

        self.model = torch.nn.Sequential(*model)

    def forward(self, input_data):
        if self.outermost or self.noskip:
            return self.model(input_data)

        return torch.cat([input_data, self.model(input_data)], 1)
