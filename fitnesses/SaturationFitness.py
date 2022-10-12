import torch

from .fitness_interface import FitnessInterface


class SaturationFitness(FitnessInterface):
    def __init__(self):
        super(SaturationFitness, self).__init__()

        self.saturation_weight = -1

    def evaluate(self, img):
        _pixels = img.permute(0, 2, 3, 1).reshape(-1, 3)
        rg = _pixels[:, 0] - _pixels[:, 1]
        yb = 0.5 * (_pixels[:, 0] + _pixels[:, 1]) - _pixels[:, 2]
        rg_std, rg_mean = torch.std_mean(rg)
        yb_std, yb_mean = torch.std_mean(yb)
        std_rggb = torch.sqrt(rg_std ** 2 + yb_std ** 2)
        mean_rggb = torch.sqrt(rg_mean ** 2 + yb_mean ** 2)
        colorfullness = std_rggb + .3 * mean_rggb

        return -colorfullness * self.saturation_weight / 10.0

