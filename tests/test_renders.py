import argparse

import torch
from torchvision.utils import save_image

from render import *

parser = argparse.ArgumentParser(description="Evolve to objective")

args = parser.parse_args()

args.num_lines = 24
args.img_size = 256
args.n_gens = 1
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

renders = [CharsRenderer, PylinhasRenderer, OrganicRenderer, ThinOrganicRenderer, PixelRenderer, FastPixelRenderer, VQGANRenderer, ClipDrawRenderer, BigGANRenderer, LineDrawRenderer, FFTRenderer, VDiffRenderer]

for render in renders:
    r = render(args)
    individual = r.generate_individual()
    optimizers = r.to_adam(individual)
    img = r.render()

    print(r, torch.min(img).item(), torch.max(img).item())

    save_image(img, "results/" + str(r) + ".png")
