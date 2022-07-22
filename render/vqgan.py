import math
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import requests
from taming.models import cond_transformer, vqgan

from renderinterface import RenderingInterface
from utils import map_number, Vector, perpendicular, normalize


def wget_file(url, out):
    try:
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


vqgan_config_table = {
    "imagenet_f16_1024": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml',
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
    "imagenet_f16_1024": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt',
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

        self.genotype_size = 13 * args.num_lines

        vqgan_model = 'imagenet_f16_16384'
        config_path = f'models/vqgan_{vqgan_model}.yaml'
        checkpoint_path = f'models/vqgan_{vqgan_model}.ckpt'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

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

        print(z.shape)

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.args.num_lines, self.genotype_size))

    def __str__(self):
        return "organic"

    def render(self, a, img_size):
        pass

