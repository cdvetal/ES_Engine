import torch

from es_engine import keras_fitness
from render.thinorg import ThinOrganicRenderer
from render.chars import CharsRenderer
from render.pylinhas import PylinhasRenderer
from render.organic import OrganicRenderer

from config import *
from utils import get_active_models_from_arg

renders = [ThinOrganicRenderer(), CharsRenderer(), PylinhasRenderer(), OrganicRenderer()]

for render in renders:
    values = torch.rand(10, render.genotype_size)

    img = render.render(values, 512)
    img.save("images/img_{}.png".format(render))


values = torch.rand(NUM_LINES, NUM_COLS)
r = keras_fitness(values, 256, 256)
print(r)

