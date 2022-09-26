import numpy as np
import torchvision.transforms.functional as TF

from render.biggan import BigGAN
from render.renderinterface import RenderingInterface
from utils import CondVectorParameters


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        output_size = args.img_size if args.img_size in [128, 256, 512] else 256

        self.model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
        self.model.to(self.device).eval()

        self.num_latents = len(self.model.config.layers) + 1

        self.genotype_size = (self.num_latents * 256)
        self.real_genotype_size = self.genotype_size

    def chunks(self, array):
        return np.reshape(array, (self.num_latents, 256))

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self, a, cur_iteration):
        conditional_vector = CondVectorParameters(a, num_latents=self.num_latents).to(self.device)

        cond_vector = conditional_vector()
        out = self.model(cond_vector, 1)
        out = TF.to_pil_image(out.squeeze())

        return out


