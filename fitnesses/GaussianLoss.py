import torch

from fitnesses.fitness_interface import FitnessInterface


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(ylen=256, xlen=256, stdy=128, stdx=128):
    """Returns a 2D Gaussian kernel array."""
    gkerny = gaussian_fn(ylen, std=stdy)
    gkernx = gaussian_fn(xlen, std=stdx)
    gkern2d = torch.outer(gkerny, gkernx)
    return gkern2d


class GaussianLoss(FitnessInterface):
    def __init__(self, ):
        super(GaussianLoss, self).__init__()

        self.gaussian_weight = -1
        self.gaussian_std = (40, 40)
        self.gaussian_color = (255, 255, 255)

    def evaluate(self, img):
        gaus = gkern(img.size()[2], img.size()[3], *self.gaussian_std).to(self.device)
        color = torch.zeros(img.size()).to(self.device)
        Rval, Gval, Bval = self.gaussian_color
        color[:, 0, :, :] = Rval / 255
        color[:, 1, :, :] = Gval / 255
        color[:, 2, :, :] = Bval / 255
        # mseloss = nn.MSELoss()
        loss = torch.abs(img - color)
        # print(loss.size())
        loss = loss * torch.abs(1 - gaus)

        cur_loss = torch.mean(loss)
        return cur_loss * self.gaussian_weight
