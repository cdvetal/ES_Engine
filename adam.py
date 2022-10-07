import os

from torch import optim
from torchvision.utils import save_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch

cur_iteration = 0


def calculate_fitness(fitnesses, img):
    losses = []

    for fitness in fitnesses:
        losses.append(fitness.evaluate(img))

    losses = torch.stack(losses)
    final_loss = torch.sum(losses)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return [final_loss]


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
        loss = calculate_fitness(args.fitnesses, img)

        optimizer.zero_grad()
        (-loss[0]).backward()
        optimizer.step()

        print(loss[0])

        if args.renderer_type == "vdiff" and gen >= 1:
            lr = renderer.sample_state[6][gen] / renderer.sample_state[5][gen]
            individual = renderer.makenoise(gen, individual)
            individual.requires_grad_()
            individual = [individual]
            optimizer = optim.Adam(individual, lr=min(lr * 0.001, 0.01))

        if torch.min(img) < 0.0:
            img = (img + 1) / 2

        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")
