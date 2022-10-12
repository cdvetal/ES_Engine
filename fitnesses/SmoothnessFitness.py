import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

from .fitness_interface import FitnessInterface


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class SmoothnessLoss(FitnessInterface):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

        self.smoothness_weight = -0.5
        self.smoothness_type = 'log'
        self.smoothness_gaussian_kernel = 0
        self.smoothness_gaussian_std = 1
        self.smoothness_spacing = 1
        self.smoothness_edge_order = 1

    def evaluate(self, img):
        if self.smoothness_gaussian_kernel:
            smoothing = GaussianSmoothing(3, self.smoothness_gaussian_kernel, self.smoothness_gaussian_std).to(
                self.device)
            img = smoothing(img)

        _pixels = img.permute(0, 2, 3, 1).reshape(-1, img.shape[2], 3)
        gyr, gxr = torch.gradient(_pixels[:, :, 0], spacing=self.smoothness_spacing,
                                  edge_order=self.smoothness_edge_order)
        gyg, gxg = torch.gradient(_pixels[:, :, 1], spacing=self.smoothness_spacing,
                                  edge_order=self.smoothness_edge_order)
        gyb, gxb = torch.gradient(_pixels[:, :, 2], spacing=self.smoothness_spacing,
                                  edge_order=self.smoothness_edge_order)
        sharpness = torch.sqrt(gyr ** 2 + gxr ** 2 + gyg ** 2 + gxg ** 2 + gyb ** 2 + gxb ** 2)
        if self.smoothness_type == 'clipped':
            sharpness = torch.clamp(sharpness, max=0.5)
        elif self.smoothness_type == 'log':
            sharpness = torch.log(torch.ones_like(sharpness) + sharpness)
        sharpness = torch.mean(sharpness)

        return sharpness * self.smoothness_weight
