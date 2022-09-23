import numpy as np
import torch
import torchvision.transforms.functional as TF

from render.biggan import BigGAN
from render.renderinterface import RenderingInterface


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        output_size = args.img_size if args.img_size in [128, 256, 512] else 256

        self.model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
        self.model.to(self.device).eval()

        self.genotype_size = (16 * 256)
        self.real_genotype_size = self.genotype_size

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (16, 256))

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self, a, cur_iteration):
        a = torch.tensor(a).float().to(self.device)
        out = self.model(a, 1)

        out = TF.to_pil_image(out.squeeze())

        return out


