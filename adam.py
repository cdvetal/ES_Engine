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
    individual = renderer.to_adam(individual)

    optimizer = optim.Adam(individual, lr=args.lr)

    for gen in range(args.n_gens):
        print("Generation:", gen)
        cur_iteration = gen

        img = renderer.render(individual)
        fitness = calculate_fitness(args.fitnesses, img)

        optimizer.zero_grad()
        (-fitness).backward()
        optimizer.step()

        print(fitness)

        if args.renderer_type == "vdiff" and gen >= 1:
            lr = renderer.sample_state[6][gen] / renderer.sample_state[5][gen]
            individual = renderer.makenoise(gen, individual)
            individual.requires_grad_()
            individual = [individual]
            optimizer = optim.Adam(individual, lr=min(lr * 0.001, 0.01))

        if torch.min(img) < 0.0:
            img = (img + 1) / 2

        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")
