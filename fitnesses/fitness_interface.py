import subprocess

import torch


class FitnessInterface:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, img):
        pass


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2
    # n = max(start1, min(stop1, n))
    # n = ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2
    # return n


def calculate_fitness(fitnesses, img):
    losses = []

    for fitness in fitnesses:
        losses.append(fitness.evaluate(img))

    losses = torch.stack(losses)
    final_loss = torch.sum(losses)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return final_loss
