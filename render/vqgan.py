import subprocess

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from taming.models import vqgan
import torchvision

from render.renderinterface import RenderingInterface
from utils import create_save_folder


def wget_file(url, out):
    try:
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


vqgan_config_table = {
    "imagenet_f16_1024": 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
    "imagenet_f16_16384": 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
    "imagenet_f16_16384m": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml',
    "openimages_f16_8192": 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
    "coco": 'https://dl.nmkd.de/ai/clip/coco/coco.yaml',
    "faceshq": 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT',
    "wikiart_1024": 'http://mirror.io.community/blob/vqgan/wikiart.yaml',
    "wikiart_16384": 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml',
    "wikiart_16384m": 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml',
    "sflckr": 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1',
}

vqgan_checkpoint_table = {
    "imagenet_f16_1024": 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
    "imagenet_f16_16384": 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
    "imagenet_f16_16384m": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt',
    "openimages_f16_8192": 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
    "coco": 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt',
    "faceshq": 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt',
    "wikiart_1024": 'http://mirror.io.community/blob/vqgan/wikiart.ckpt',
    "wikiart_16384": 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt',
    "wikiart_16384m": 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt',
    "sflckr": 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'
}


class VQGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(VQGANRenderer, self).__init__(args)

        main_path = 'models'
        vqgan_model = 'imagenet_f16_1024'
        config_path = f'{main_path}/vqgan_{vqgan_model}.yaml'
        checkpoint_path = f'{main_path}/vqgan_{vqgan_model}.ckpt'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        create_save_folder(main_path, '')

        wget_file(vqgan_config_table[vqgan_model], config_path)
        wget_file(vqgan_checkpoint_table[vqgan_model], checkpoint_path)

        self.config = OmegaConf.load(config_path)

        self.model = vqgan.VQModel(**self.config.model.params)
        self.model.eval().requires_grad_(False)
        self.model.init_from_ckpt(checkpoint_path)
        del self.model.loss
        self.model = self.model.to(self.device)

        e_dim = self.model.quantize.e_dim
        n_toks = self.model.quantize.n_e
        z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        num_resolutions = self.model.decoder.num_resolutions
        f = 2 ** (num_resolutions - 1)
        image_size = 224
        toksX, toksY = image_size // f, image_size // f

        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks).float()
        z = one_hot @ self.model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_shape = z.shape

        self.genotype_size = torch.numel(z)
        self.real_genotype_size = self.genotype_size

        print(self.z_shape, self.genotype_size)

        self.replace_grad = ReplaceGrad.apply
        self.clamp_with_grad = ClampWithGrad.apply

        self.to_pil = torchvision.transforms.ToPILImage()

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, self.z_shape)

    def vector_quantize(self, x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return self.replace_grad(x_q, x)

    def __str__(self):
        return "VQGAN"

    def render(self, a, img_size, cur_iteration):
        z_q = self.vector_quantize(a.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        out = self.clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

        return self.to_pil(out.squeeze(0))

