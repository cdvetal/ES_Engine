import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from render.renderinterface import RenderingInterface


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


class FastPixelRenderer(RenderingInterface):
    def __init__(self, args):
        super(FastPixelRenderer, self).__init__(args)

        self.num_lines = args.num_lines
        self.img_size = args.img_size
        self.device = args.device

        self.num_cols = self.num_lines
        self.num_rows = self.num_lines

        self.pixel_size = tuple([self.num_rows, self.num_cols])

        self.genotype_size = 3
        self.real_genotype_size = self.genotype_size * (args.num_lines * args.num_lines)

    def chunks(self, array):
        array = torch.tensor(array, dtype=torch.float)
        return array.view(self.num_lines, self.num_lines, self.genotype_size)

    def __str__(self):
        return "fastpixel"

    def render(self, a, cur_iteration):
        a = torch.tensor(a).float().to(self.device)

        # out = F.interpolate((a + 1) / 2, size=self.pixel_size, mode="bilinear", align_corners=False)
        out = F.interpolate(a, size=(self.img_size, self.img_size), mode="nearest")

        print(out.shape)

        return None
