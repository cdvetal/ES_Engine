import subprocess

import torch


class FitnessInterface:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, img):
        pass


def calculate_fitness(fitnesses, img):
    losses = []

    for fitness in fitnesses:
        losses.append(fitness.evaluate(img))

    losses = torch.stack(losses)
    final_loss = torch.sum(losses)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return final_loss
