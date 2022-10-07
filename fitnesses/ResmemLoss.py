import os

import resmem
import torch
from resmem import ResMem
from torchvision import transforms

from .fitness_interface import FitnessInterface
from utils import wget_file, map_number

resmem_url = 'https://github.com/pixray/resmem/releases/download/1.1.3_model/model.pt'


recenter = transforms.Compose((
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    )
)


class ResmemLoss(FitnessInterface):
    def __init__(self):
        super(ResmemLoss, self).__init__()
        # make sure resmem has model file
        if not os.path.exists(resmem.path):
            wget_file(resmem_url, resmem.path)

        self.model = ResMem(pretrained=True).to(self.device)
        # Set the model to inference mode.
        self.model.eval()

        self.symmetry_weight = -1

    def evaluate(self, img):
        # print(images.shape)
        image_x = recenter(img)
        # print(image_x.shape)

        prediction = self.model(image_x)
        # print(prediction)
        # print(prediction.shape)
        mean = torch.mean(prediction)
        # loss seems to bottom out at 0.4? ¯\_(ツ)_/¯
        mapped_mean = map_number(mean, 0.4, 1.0, 0, 1)
        the_loss = 0.05 * mapped_mean

        return the_loss
