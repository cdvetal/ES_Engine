import math
import os

import cairo
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from v_diffusion_pytorch.diffusion import get_model, get_models, sampling, utils

from render.renderinterface import RenderingInterface
from utils import map_number
from utils import wget_file
from v_diffusion_pytorch.diffusion.utils import get_spliced_ddpm_cosine_schedule

model_urls = {
    "yfcc_2": "https://the-eye.eu/public/AI/models/v-diffusion/yfcc_2.pth",
    "yfcc_1": "https://the-eye.eu/public/AI/models/v-diffusion/yfcc_1.pth",
    "cc12m_1": "https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1.pth",
    "cc12m_1_cfg": "https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth",
    "danbooru_128": "https://the-eye.eu/public/AI/models/v-diffusion/danbooru_128.pth",
    "imagenet_128": "https://the-eye.eu/public/AI/models/v-diffusion/imagenet_128.pth",
    "wikiart_128": "https://the-eye.eu/public/AI/models/v-diffusion/wikiart_128.pth",
    "wikiart_256": "https://the-eye.eu/public/AI/models/v-diffusion/wikiart_256.pth",
}


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


clamp_with_grad = ClampWithGrad.apply

ROUNDUP_SIZE = 128
# https://stackoverflow.com/a/8866125/1010653


def roundup(x, n):
    return int(math.ceil(x / float(n))) * n


class VDiffRenderer(RenderingInterface):
    def __init__(self, args):
        super(VDiffRenderer, self).__init__(args)

        self.vdiff_model = 'yfcc_1'
        self.canvas_width = args.img_size
        self.canvas_height = args.img_size
        self.gen_width = roundup(self.canvas_width, ROUNDUP_SIZE)
        self.gen_height = roundup(self.canvas_height, ROUNDUP_SIZE)
        self.iterations = args.n_gens
        self.eta = 1
        self.vdiff_skip = 0

        model = get_model(self.vdiff_model)()
        checkpoint = f'models/{self.vdiff_model}.pth'

        if not (os.path.exists(checkpoint) and os.path.isfile(checkpoint)):
            wget_file(model_urls[self.vdiff_model], checkpoint)

        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        if args.device.type == 'cuda':
            model = model.half()
        model = model.to(args.device).eval().requires_grad_(False)

        self.model = model
        self.device = args.device
        self.pred = None
        self.v = None

        self.iter = 0

        # compute self.t based on vdiff_skip
        top_val = map_number(self.vdiff_skip, 0, 100, 1, 0)
        # print("Using a max for vdiff skip of ", top_val)
        self.t = torch.linspace(top_val, 0, self.iterations + 2, device=self.device)[:-1]
        # print("self.t is ", self.t.shape)

        self.x = torch.randn([1, 3, self.gen_height, self.gen_width], device=self.device)

        self.steps = get_spliced_ddpm_cosine_schedule(self.t)

        # [model, steps, eta, extra_args, ts, alphas, sigmas]
        self.sample_state = sampling.sample_setup(self.model, self.x, self.steps, self.eta, {})

        self.genotype_size = 1 * 3 * self.gen_height * self.gen_width
        self.real_genotype_size = self.genotype_size

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (1, 3, self.gen_height, self.gen_width))

    def __str__(self):
        return "vdiff"

    def render(self, a, cur_iteration):
        x = torch.tensor(a).to(self.device)

        # sample_state = sampling.sample_setup(self.model, x, self.steps, self.eta, {})

        pred, v, next_x = sampling.sample_step(self.sample_state, x, cur_iteration, self.pred, self.v)
        self.pred = pred.detach()
        self.v = v.detach()
        pixels = clamp_with_grad(pred.add(1).div(2), 0, 1)

        # center crop
        margin_x = int((self.gen_width - self.canvas_width) / 2)
        margin_y = int((self.gen_height - self.canvas_height) / 2)
        if margin_x != 0 or margin_y != 0:
            pixels = pixels[:, :, margin_y:(margin_y + self.canvas_height), margin_x:(margin_x + self.canvas_width)]

        # save a copy for the next iteration
        return TF.to_pil_image(pixels[0].cpu())

