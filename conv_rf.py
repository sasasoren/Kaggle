import math
import numpy as np
import torch
import torch.nn as nn
from random import sample
import torch.nn.functional as F


class ConvRF(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            padding=0,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            num_kernels=1,
            kernels_path=None,
            ):
        """
        This convolution class replaces the learnable convolutional kernels (1D, 2D, or 3D) with linear combination of
        fixed predesigned kernels. By Default we use nikos_kernels, a complete family of sparse directional filters.

        :param kernels_path: str, absolute path to our fixed kernels numpy array with either of the following shapes:
        shape = (num_filters, length), or (num_filters, height, width) or (num_filters, depth, height, width)
        :param num_kernels: int, number of (randomly selected) predesigned kernels in every convolutional channel
        (which is either an in_channel or an out_channel)
        """
        super(ConvRF, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.kernels_path = kernels_path

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)


class Conv2dRF(ConvRF):

    def __init__(self,
                 in_channels,
                 out_channels,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 num_kernels=1,
                 kernels_path=None,
                 ):
        super(Conv2dRF, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_kernels=num_kernels,
            kernels_path=kernels_path,
        )
        self.kernels = np.load(self.kernels_path)
        assert self.kernels.ndim == 3
        self.total_kernels, self.kernel_size = self.kernels.shape[0], (self.kernels.shape[1], self.kernels.shape[2])
        assert 1 <= self.num_kernels <= self.total_kernels

        lin_comb_kernels = torch.Tensor(self.build_kernels(
            self.kernels,
            self.num_kernels,
            self.out_channels,
            self.in_channels))
        self.register_buffer("lin_comb_kernels", lin_comb_kernels)
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, self.num_kernels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def build_kernels(self, kernels, num_kernels, out_channels, in_channels):
        """
        :param kernels:  np.array, fixed kernels of shape [num_kernels, kernel_height, kernel_width]
        :param num_kernels: int, number of randomly selected subset of kernels for each channel (in_channel or
        out_channel)
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return:
        """
        assert kernels.ndim == 3
        total_kernels, h, w = kernels.shape[0], kernels.shape[1], kernels.shape[2]
        lin_comb_kernels = np.zeros((out_channels, in_channels, num_kernels, h, w), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels):
                lin_comb_kernels[k, j] = kernels[sample(range(0, total_kernels), num_kernels), :, :]
        return lin_comb_kernels

    def forward(self, x):

        out = F.conv2d(
            input=x,
            weight=torch.einsum("ijk, ijklm -> ijlm", self.weight, self.lin_comb_kernels),  # lin comb op
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return out


class Conv3dRF(ConvRF):

    def __init__(self,
                 in_channels,
                 out_channels,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 num_kernels=1,
                 kernels_path=None,
                 ):
        super(Conv3dRF, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_kernels=num_kernels,
            kernels_path=kernels_path,
        )

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        self.kernels = np.load(kernels_path)
        assert self.kernels.ndim == 4
        self.total_kernels, self.kernel_size = \
            self.kernels.shape[0], (self.kernels.shape[1], self.kernels.shape[2], self.kernels.shape[3])
        assert 1 <= self.num_kernels <= self.total_kernels

        lin_comb_kernels = torch.Tensor(self.build_fixed_kernels(
            self.kernels,
            self.num_kernels,
            self.out_channels,
            self.in_channels))
        self.register_buffer("lin_comb_kernels", lin_comb_kernels)

        # self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.num_kernels, 1, 1))
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.num_kernels))
        self.reset_parameters()

    def reset_parameters(self, ):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # # xavier initializer
        # nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     stdv = 1. / math.sqrt(self.in_channels)
        #     self.bias.data.uniform_(-stdv, stdv)

    def build_fixed_kernels(self, kernels, num_kernels, out_channels, in_channels):
        """
        :param kernels:  np.array, fixed kernels of shape [num_kernels, kernel_depth, kernel_height, kernel_width]
        :param num_kernels: int, number of randomly selected subset of kernels for each channel (in_channel or
        out_channel)
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return:
        """
        total_kernels, d, h, w = kernels.shape[0], kernels.shape[1], kernels.shape[2], kernels.shape[3]
        lin_comb_kernels = np.zeros((out_channels, in_channels, num_kernels, d, h, w), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels):
                lin_comb_kernels[k, j] = kernels[sample(range(0, total_kernels), num_kernels), :, :, :]
        return lin_comb_kernels

    def forward(self, x):
        # lin_comb_op = sum(torch.mul(self.weight, filters), dim=2, keepdim=False)
        return F.conv3d(
            input=x,
            weight=torch.einsum("ijk, ijklmn -> ijlmn", self.weight, self.lin_comb_kernels),  # lin comb op
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)