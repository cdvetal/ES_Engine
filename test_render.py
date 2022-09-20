import torch

from render import *

from es_engine import setup_args
from render.vdiff import VDiffRenderer

args = setup_args()

# renders = [ThinOrganicRenderer(args), CharsRenderer(args), PylinhasRenderer(args), OrganicRenderer(args), ClipDrawRenderer(args), LineRenderer(args)]
renders = [VDiffRenderer(args)]

for render in renders:
    values = torch.rand(render.real_genotype_size)

    values = render.chunks(values)
    img = render.render(values, 512)
    img.save("images/img_{}.png".format(render))
