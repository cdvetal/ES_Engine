import torch
from torch import nn

from fitnesses.fitness_interface import FitnessInterface


class SymmetryLoss(FitnessInterface):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

        self.symmetry_weight = -1

    def evaluate(self, img):
        mseloss = nn.MSELoss()
        cur_loss = mseloss(img, torch.flip(img, [3]))
        return cur_loss * self.symmetry_weight

