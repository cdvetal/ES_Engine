import numpy as np
import torch
from aphantasia.image import to_valid_rgb, fft_image

from render.renderinterface import RenderingInterface


# TODO CMA-ES - MemoryError: Unable to allocate 4.54 TiB for an array with shape (789504, 789504) and data type float64

class FFTRenderer(RenderingInterface):
    def __init__(self, args):
        super(FFTRenderer, self).__init__(args)

        self.img_size = args.img_size

        self.fft_use = "fft"

        self.decay = 1.5

        self.device = args.device

    def chunks(self, array):
        return np.reshape(array, self.shape)

    def generate_individual(self):
        shape = [1, 3, self.img_size, self.img_size]

        params, image_f, sz = fft_image(shape, sd=0.01, decay_power=self.decay)

        for param in params:
            print(param.shape)

        individual = params[0].cpu().detach().numpy()

        self.shape = individual.shape

        return individual.flatten()

    def to_adam(self, individual):
        ind_copy = np.copy(individual)

        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)

        shape = [1, 3, self.img_size, self.img_size]

        params, image_f, sz = fft_image(shape, sd=0.01, decay_power=self.decay, resume=ind_copy)

        self.image_f = image_f

        return params

    def __str__(self):
        return "fftdrawer"

    def render(self, input_ind):
        image_f = to_valid_rgb(self.image_f, colors=1.5)
        img = image_f(contrast=0.9)
        return img
