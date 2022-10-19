import numpy as np
import torch
from aphantasia.image import to_valid_rgb, fft_image, dwt_image

from render.renderinterface import RenderingInterface


# TODO CMA-ES - MemoryError: Unable to allocate 4.54 TiB for an array with shape (789504, 789504) and data type float64

class FFTRenderer(RenderingInterface):
    def __init__(self, args):
        super(FFTRenderer, self).__init__(args)

        self.img_size = args.img_size

        self.fft_use = "fft"

        self.decay = 1.5

        self.device = args.device

        self.individual = None
        self.image_f = None
        self.shape = None

    def chunks(self, array):
        return np.reshape(array, self.shape)

    def generate_individual(self):
        shape = [1, 3, self.img_size, self.img_size]

        wave = 'coif2'
        sharp = 0.3
        colors = 1.5
        params, image_f, sz = dwt_image(shape, wave, sharp, colors)

        # for param in params:
        #     print(param.shape)

        self.params = params
        self.image_f = image_f

        return None

    def to_adam(self, individual, gradients=True):
        """
        self.individual = np.copy(individual)

        self.individual = self.chunks(self.individual)
        self.individual = torch.tensor(self.individual).float().to(self.device)

        shape = [1, 3, self.img_size, self.img_size]

        # params, image_f, sz = fft_image(shape, sd=0.01, decay_power=self.decay, resume=self.individual)
        wave = 'coif2'
        sharp = 0.3
        colors = 1.5
        params, image_f, sz = dwt_image(shape, wave, sharp, colors, resume=self.individual)

        self.image_f = image_f

        optimizer = torch.optim.Adam(params, lr=0.3)
        """
        optimizer = torch.optim.Adam(self.params, lr=0.3)

        return [optimizer]

    def __str__(self):
        return "fftdrawer"

    def render(self):
        image_f = to_valid_rgb(self.image_f, colors=1.5)
        img = image_f(contrast=0.9)
        return img
