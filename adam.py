import os

from torch import optim
from torchvision.utils import save_image

from fitnesses import calculate_fitness


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch

cur_iteration = 0


def main_adam(args):
    global cur_iteration

    renderer = args.renderer

    individual = renderer.generate_individual()
    optimizers = renderer.to_adam(individual)

    for gen in range(args.n_gens):
        print("Generation:", gen)
        cur_iteration = gen

        img = renderer.render()
        fitness = calculate_fitness(args.fitnesses, img, args.normalization)

        for optimizer in optimizers:
            optimizer.zero_grad()

        (-fitness).backward()

        for optimizer in optimizers:
            optimizer.step()

        print(fitness.item())

        if args.renderer_type == "vdiff":
            # optimizers = renderer.to_adam(individual, gen=gen)
            lr = renderer.sample_state[6][gen] / renderer.sample_state[5][gen]
            renderer.individual = renderer.makenoise(gen)
            renderer.individual.requires_grad_()
            to_optimize = [renderer.individual]
            opt = optim.Adam(to_optimize, lr=min(lr * 0.001, 0.01))
            optimizers = [opt]

        if torch.min(img) < 0.0:
            img = (img + 1) / 2

        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")
