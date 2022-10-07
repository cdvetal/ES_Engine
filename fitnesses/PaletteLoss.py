import torch

from fitnesses.fitness_interface import FitnessInterface


class PaletteLoss(FitnessInterface):
    def __init__(self, palette):
        super(PaletteLoss, self).__init__()

        self.palette_weight = -1
        self.target_palette = torch.FloatTensor(palette).requires_grad_(False).to(self.device)

    def evaluate(self, img):
        _pixels = img.permute(0, 2, 3, 1).reshape(-1, 3)
        palette_dists = torch.cdist(self.target_palette, _pixels, p=2)
        best_guesses = palette_dists.argmin(axis=0)
        diffs = _pixels - self.target_palette[best_guesses]
        palette_loss = torch.mean(torch.norm(diffs, 2, dim=1)) * img.shape[0]
        return palette_loss * self.palette_weight / 10.0
