import torch

from render import *

from es_engine import setup_args

args = setup_args()

renders = [ThinOrganicRenderer(args), CharsRenderer(args), PylinhasRenderer(args), OrganicRenderer(args), ClipDrawRenderer(args), LineRenderer(args)]

for render in renders:
    values = torch.rand(1, render.real_genotype_size)

    img = render.render(values, 512)
    img.save("images/img_{}.png".format(render))
