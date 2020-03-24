import torch.nn as nn
import conv_rf

nonlineartity_names = \
    [
        "Tanh",
        "Tanhshrink",
        "Hardtanh",
        "Sigmoid",
        "Hardshrink",
        "Softshrink",
        "Softsign",
        "Softplus",
        "ReLU",
        "LeakyReLU",
        "ELU",
        "SELU",
        "CELU",
        "PReLU"
    ]


class NonLinearity(nn.Module):
    def __init__(self, nonlin_name=None):
        super(NonLinearity, self).__init__()
        self.nonlin_name = nonlin_name
        if self.nonlin_name is not None:
            self.nonlinearity = getattr(nn, self.nonlin_name)()
            # print("nonlinearity: ", self.nonlinearity)

    def forward(self, x):
        if self.nonlin_name is None:
            return x
        else:
            return self.nonlinearity(x)


class BasicConv(nn.Module):

    def __init__(self, conv_model, in_channels, out_channels, kernel_size,
                 bias=True, bn=None, nonlin_name=None, **kwargs):
        super(BasicConv, self).__init__()

        if hasattr(nn, conv_model):
            if 'num_kernels' in kwargs:
                del kwargs['num_kernels']
            if 'kernels_path' in kwargs:
                del kwargs['kernels_path']
            cnn_model = getattr(nn, conv_model)
            self.conv = cnn_model(in_channels, out_channels, kernel_size, bias=bias, **kwargs)

        elif hasattr(conv_rf, conv_model):
            cnn_model = getattr(conv_rf, conv_model)
            # kernel size is inferred from kernels_path
            self.conv = cnn_model(in_channels, out_channels, bias=bias, **kwargs)

        else:
            ValueError("Not implemented")
        # print("kwargs after: ", kwargs)

        self.bn = bn
        if self.bn is not None:
            self.bn_layer = self.bn(out_channels, eps=0.001)
        self.nonlin = NonLinearity(nonlin_name)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn_layer(x)
        return self.nonlin(x)
