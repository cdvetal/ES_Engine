import numpy as np
import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names)
from torchvision import transforms

from render.renderinterface import RenderingInterface


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        self.num_lines = args.num_lines

        self.model = BigGAN.from_pretrained('biggan-deep-512').to(self.device)

        # self.genotype_size = self.model.
        # self.real_genotype_size = self.genotype_size * (args.num_lines * args.num_lines)

        # Prepare a input
        self.truncation = 0.4
        class_vector = one_hot_from_names([args.target_class], batch_size=1)
        # noise_vector = truncated_noise_sample(truncation=0.4, batch_size=1)

        # All in tensors
        # noise_vector = torch.from_numpy(noise_vector)
        # print(noise_vector.shape)
        self.class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        # noise_vector = noise_vector.to('cuda')
        self.class_vector = self.class_vector.to(args.device)
        # model.to('cuda')

        # self.genotype_size = self.model.z_dim
        # self.real_genotype_size = self.genotype_size
        self.genotype_size = 128
        self.real_genotype_size = 128

    def chunks(self, array):
        array = np.array(array)
        return np.reshape(array, (1, self.genotype_size))

    def __str__(self):
        return "biggan"

    def render(self, a, img_size, cur_iteration):
        noise_vector = torch.from_numpy(a).to(self.device).float()
        # Generate an image
        with torch.no_grad():
            output = self.model(noise_vector, self.class_vector, self.truncation)

        # If you have a GPU put back on CPU
        output = output.to('cpu').squeeze()
        transform = transforms.ToPILImage()
        output = transform(output)
        return output

