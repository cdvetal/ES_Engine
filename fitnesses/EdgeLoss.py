import torch
from torch import nn

from fitnesses.fitness_interface import FitnessInterface
from utils import map_number


class EdgeLoss(FitnessInterface):
    def __init__(self):
        super(EdgeLoss, self).__init__()

        self.edge_weight = -1

        self.edge_thickness = 5
        self.edge_margins = None
        self.edge_color = [1., 1., 1.]
        self.edge_color_weight = 0.1
        self.global_color_weight = 0.05

        if self.edge_margins is None:
            t = self.edge_thickness
            self.edge_margins = (t, t, t, t)

    def evaluate(self, img):
        zers = torch.zeros(img.size()).to(self.device)
        Rval, Gval, Bval = self.edge_color
        zers[:, 0, :, :] = Rval
        zers[:, 1, :, :] = Gval
        zers[:, 2, :, :] = Bval
        cur_loss = torch.tensor(0.0).cuda()
        mseloss = nn.MSELoss()
        lmax = img.size()[2]
        rmax = img.size()[3]
        left, right, upper, lower = self.edge_margins
        left = int(map_number(left, 0, 100, 0, rmax))
        right = int(map_number(right, 0, 100, 0, rmax))
        upper = int(map_number(upper, 0, 100, 0, lmax))
        lower = int(map_number(lower, 0, 100, 0, lmax))
        # print(left, right, upper, lower)
        lloss = mseloss(img[:, :, :, :left], zers[:, :, :, :left])
        rloss = mseloss(img[:, :, :, rmax - right:], zers[:, :, :, rmax - right:])
        uloss = mseloss(img[:, :, :upper, left:rmax - right], zers[:, :, :upper, left:rmax - right])
        dloss = mseloss(img[:, :, lmax - lower:, left:rmax - right], zers[:, :, lmax - lower:, left:rmax - right])
        if left != 0:
            cur_loss += lloss
        if right != 0:
            cur_loss += rloss
        if upper != 0:
            cur_loss += uloss
        if lower != 0:
            cur_loss += dloss
        if self.global_color_weight:
            gloss = mseloss(img[:, :, :, :], zers[:, :, :, :]) * self.global_color_weight
            cur_loss += gloss
        cur_loss *= self.edge_color_weight
        return cur_loss


