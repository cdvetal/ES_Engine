import numpy as np
import torch
from torch.nn import functional as F

from render.renderinterface import RenderingInterface


class FastPixelRenderer(RenderingInterface):
    def __init__(self, args):
        super(FastPixelRenderer, self).__init__(args)

        self.device = args.device

        self.img_size = args.img_size

        self.num_cols, self.num_rows = [args.num_lines, args.num_lines]

        self.lr = 0.1

        self.individual = None

        self.pixel_size = tuple([self.num_rows, self.num_cols])

    def chunks(self, array):
        return np.reshape(array, (3, self.num_cols, self.num_rows))

    def generate_individual(self):
        individual = np.random.rand(3, self.num_cols, self.num_rows)
        return individual.flatten()

    def get_individual(self):
        return None

    def to_adam(self, individual, gradients=True):
        self.individual = np.copy(individual)

        self.individual = self.chunks(self.individual)
        self.individual = torch.tensor(self.individual).float().unsqueeze(0).to(self.device)

        self.individual = F.interpolate((self.individual + 1) / 2, size=self.pixel_size, mode="bilinear", align_corners=False)

        if gradients:
            self.individual.requires_grad = True

        optimizer = torch.optim.Adam([self.individual], lr=0.1)

        return [optimizer]

    def __str__(self):
        return "fastpixeldraw"

    def render(self):
        img = F.interpolate(self.individual, size=(self.img_size, self.img_size), mode="nearest")
        return img
