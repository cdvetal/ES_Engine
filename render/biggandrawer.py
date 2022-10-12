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

        self.genotype_size = (self.num_latents * 256)
        self.real_genotype_size = self.genotype_size

        self.x = None

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

    def get_individual(self, adam_ind):
        return adam_ind[0].cpu().detach().numpy().flatten()

    def to_adam(self, individual):
        ind_copy = np.copy(individual)
        ind_copy = self.chunks(ind_copy)
        ind_copy = torch.tensor(ind_copy).float().to(self.device)
        ind_copy.requires_grad = True
        return [ind_copy]

    def chunks(self, array):
        # if type(array) is list:
        #     array = torch.tensor(array).float()
        # return array.view(self.num_latents, 256)
        return np.reshape(array, (self.num_latents, 256))

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self, input_ind):
        out = self.model(input_ind[0], 1)

        return out
