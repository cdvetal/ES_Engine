import subprocess
from pathlib import Path

import clip
import torch
import torchvision
from torch import nn

from .fitness_interface import FitnessInterface


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# if you changed the MLP architecture during training, change it also here:
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class Aesthetic2Fitness(FitnessInterface):
    def __init__(self, model=None, preprocess=None, clip_model="ViT-L/14"):
        super(Aesthetic2Fitness, self).__init__()
        self.aesthetic_target = 1
        # Available here: https://github.com/christophschuhmann/improved-aesthetic-predictor
        self.model_path = Path("models/sac+logos+ava1-l14-linearMSE.pth")

        if not self.model_path.exists():
            wget_file("https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac%2Blogos%2Bava1-l14-linearMSE.pth", self.model_path)

        self.mlp = MLP(768).to(self.device)
        self.mlp.load_state_dict(torch.load(self.model_path))
        self.target_rating = torch.ones(size=(1, 1)) * self.aesthetic_target
        self.target_rating = self.target_rating.to(self.device)

        self.clip_model = clip_model

        if model is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.FloatTensor

    def evaluate(self, img, normalization=False):
        img = img.to(self.device)

        """
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
        """

        img = torchvision.transforms.functional.resize(img, (224, 224))

        image_features = self.model.encode_image(img)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())

        aes_rating = self.mlp(torch.from_numpy(im_emb_arr).to(self.device).type(self.tensor))
        aes_rating = aes_rating.square().mean() * 0.02

        return aes_rating
