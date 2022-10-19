import subprocess
from pathlib import Path

import clip
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from .fitness_interface import FitnessInterface


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class AestheticFitness(FitnessInterface):
    def __init__(self, model=None, preprocess=None, clip_model="ViT-B/32"):
        super(AestheticFitness, self).__init__()
        self.aesthetic_target = 1
        # Only available here: https://twitter.com/RiversHaveWings/status/1472346186728173568
        self.model_path = Path("models/ava_vit_b_16_linear.pth")

        if not self.model_path.exists():
            wget_file("https://cdn.discordapp.com/attachments/821173872111517696/921905064333967420/ava_vit_b_16_linear.pth", self.model_path)

        layer_weights = torch.load(self.model_path)
        self.ae_reg = nn.Linear(512, 1).to(self.device)
        # self.ae_reg.load_state_dict(torch.load(self.model_path))
        self.ae_reg.bias.data = layer_weights["bias"].to(self.device)
        self.ae_reg.weight.data = layer_weights["weight"].to(self.device)
        self.target_rating = torch.ones(size=(1, 1)) * self.aesthetic_target
        self.target_rating = self.target_rating.to(self.device)

        self.aesthetic_weight = -1

        self.clip_model = "ViT-B/32"

        if model is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

    def evaluate(self, img, normalization=False):
        p_s = []

        _, channels, sideX, sideY = img.shape
        for ch in range(32):  # TODO - Maybe change here
            size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
            offsetx = torch.randint(0, sideX - size, ())
            offsety = torch.randint(0, sideX - size, ())
            apper = img[:, :, offsetx:offsetx + size, offsety:offsety + size]
            p_s.append(torch.nn.functional.interpolate(apper, (224, 224), mode='nearest'))
        # convert_tensor = torchvision.transforms.ToTensor()
        into = torch.cat(p_s, 0).to(self.device)

        if normalization:
            normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711))
            into = normalize((into + 1) / 2)

        image_features = self.model.encode_image(into)

        aes_rating = self.ae_reg(F.normalize(image_features.float(), dim=-1)).to(self.device)
        aes_distance = (aes_rating - self.target_rating).square().mean() * 0.02

        aes_fitness = aes_distance * self.aesthetic_weight

        return aes_fitness
