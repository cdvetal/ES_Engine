import numpy as np
import torch
import torchvision.transforms.functional as TF

from render.biggan import BigGAN
from render.renderinterface import RenderingInterface


class CondVectorParameters(torch.nn.Module):
    def __init__(self, ind_numpy, num_latents=15):
        super(CondVectorParameters, self).__init__()
        reshape_array = ind_numpy.reshape(num_latents, -1)
        self.normu = torch.nn.Parameter(torch.tensor(reshape_array).float())
        self.thrsh_lat = torch.tensor(1)
        self.thrsh_cls = torch.tensor(1.9)

    #  def forward(self):
    # return self.ff2(self.ff1(self.latent_code)), torch.softmax(1000*self.ff4(self.ff3(self.cls)), -1)
    #   return self.normu, torch.sigmoid(self.cls)

    # def forward(self):
    #     global CCOUNT
    #     if (CCOUNT < -10):
    #         self.normu,self.cls = copiado(self.normu, self.cls)
    #     if (MAX_CLASSES > 0):
    #         classes = differentiable_topk(self.cls, MAX_CLASSES)
    #         return self.normu, classes
    #     else:
    #         return self.normu#, torch.sigmoid(self.cls)
    def forward(self):
        return self.normu


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        output_size = args.img_size if args.img_size in [128, 256, 512] else 256

        self.model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
        self.model.to(self.device).eval()

        self.num_latents = len(self.model.config.layers) + 1

        self.genotype_size = (16 * 256)
        self.real_genotype_size = self.genotype_size

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (16, 256))

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self, a, cur_iteration):
        conditional_vector = CondVectorParameters(a, num_latents=self.num_latents).to(self.device)

        cond_vector = conditional_vector()
        out = self.model(cond_vector, 1)

        out = TF.to_pil_image(out.squeeze())

        return out


