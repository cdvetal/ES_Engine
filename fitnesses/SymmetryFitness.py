import torch
from torch import nn

from .fitness_interface import FitnessInterface


class SymmetryFitness(FitnessInterface):
    def __init__(self):
        super(SymmetryFitness, self).__init__()

        self.symmetry_weight = -1

    def evaluate(self, img, normalization=False):
        img = img.to(self.device)

        mseloss = nn.MSELoss()
        cur_loss = mseloss(img, torch.flip(img, [3]))

        # Loss must be multiplied by a negative value to obtain fitness
        symmetry_fitness = cur_loss * self.symmetry_weight / 10.0

        return symmetry_fitness

