import os
import subprocess

import resmem
import torch
from resmem import ResMem
from torchvision import transforms

from .fitness_interface import FitnessInterface

resmem_url = 'https://github.com/pixray/resmem/releases/download/1.1.3_model/model.pt'


recenter = transforms.Compose((
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    )
)


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


class ResmemFitness(FitnessInterface):
    def __init__(self):
        super(ResmemFitness, self).__init__()
        # make sure resmem has model file
        if not os.path.exists(resmem.path):
            wget_file(resmem_url, resmem.path)

        self.model = ResMem(pretrained=True).to(self.device)
        # Set the model to inference mode.
        self.model.eval()

        self.resmem_weight = -1

    def evaluate(self, img, normalization=False):
        img = img.to(self.device)

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

        # Loss must be multiplied by a negative value to obtain fitness
        resmem_fitness = the_loss * self.resmem_weight

        return resmem_fitness
