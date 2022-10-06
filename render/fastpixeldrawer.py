import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from render.renderinterface import RenderingInterface

import random

import numpy as np
import pydiffvg
import torch
from torchvision.utils import save_image

from render.renderinterface import RenderingInterface


class FastPixelRenderer(RenderingInterface):
    def __init__(self, args):
        super(FastPixelRenderer, self).__init__(args)

        self.device = args.device

        self.img_size = args.img_size

        self.num_cols, self.num_rows = [40, 40]

        self.lr = 0.1

        self.pixel_size = tuple([self.num_rows, self.num_cols])

    def chunks(self, array):
        return np.reshape(array, (3, self.num_cols, self.num_rows))

    def generate_individual(self):
        individual = np.random.rand(3, self.num_cols, self.num_rows)
        return individual.flatten()

    def get_individual(self, _):
        return None

    def to_adam(self, individual):
        ind_copy = np.copy(individual)

        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().unsqueeze(0).to(self.device)

        x = F.interpolate((ind_copy + 1) / 2, size=self.pixel_size, mode="bilinear", align_corners=False)
        x.requires_grad = True

        return [x]

    def __str__(self):
        return "fastpixeldraw"

    def render(self, input_ind):
        input_ind = input_ind[0]
        img = F.interpolate(input_ind, size=(self.img_size, self.img_size), mode="nearest")
        return img
