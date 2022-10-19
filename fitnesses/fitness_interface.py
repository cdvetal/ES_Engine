import torch


class FitnessInterface:
    """
    All these fitnesses are developed to be maximized.
    """
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, img, normalization=False):
        pass


def calculate_fitness(fitnesses, img, normalization=False):
    fitness_list = []

    for fitness in fitnesses:
        fitness_list.append(fitness.evaluate(img, normalization=normalization))

    fitness_list = torch.stack(fitness_list)
    final_fitness = torch.sum(fitness_list)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return final_fitness
