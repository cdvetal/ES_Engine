import numpy as np
import torch

from render.biggan import BigGAN
from render.renderinterface import RenderingInterface


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        self.lr = 0.03

        # Returns the closest value to the target img_size
        output_size = min([128, 256, 512], key=lambda x: abs(x - args.img_size))

        self.model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
        self.model.to(self.device).eval()

        self.num_latents = len(self.model.config.layers) + 1

        self.individual = None

    def generate_individual(self):
        z_dim = 128
        latent = torch.nn.Parameter(torch.zeros(self.num_latents, z_dim).normal_(std=1).float().cuda())
        params_other = torch.zeros(self.num_latents, 1000).normal_(-3.9, .3).cuda()
        classes = torch.sigmoid(torch.nn.Parameter(params_other))
        embed = self.model.embeddings(classes)
        cond_vector = torch.cat((latent, embed), dim=1)

        ind = cond_vector.cpu().detach().numpy().flatten()
        print(ind.shape)

        return ind

    def get_individual(self):
        return self.individual.cpu().detach().numpy().flatten()

    def to_adam(self, individual, gradients=True):
        self.individual = np.copy(individual)
        self.individual = self.chunks(self.individual)
        self.individual = torch.tensor(self.individual).float().to(self.device)

        if gradients:
            self.individual.requires_grad = True

        optimizer = torch.optim.Adam([self.individual], lr=0.04)

        return [optimizer]

    def chunks(self, array):
        # if type(array) is list:
        #     array = torch.tensor(array).float()
        # return array.view(self.num_latents, 256)
        return np.reshape(array, (self.num_latents, 256))

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self):
        out = self.model(self.individual, 1)

        return out
