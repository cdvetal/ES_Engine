import os

import clip
import torch
import torchvision
from PIL import Image

from .fitness_interface import FitnessInterface


class InputImage(FitnessInterface):
    def __init__(self, input_image, model=None, preprocess=None, clip_model="ViT-B/32"):
        super(InputImage, self).__init__()

        self.clip_model = "ViT-B/32"

        if model is None or preprocess is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

        self.input_image = input_image
        self.image_features = None

        if not os.path.exists(input_image):
            print("Image file does not exist. Ignoring..")
            self.input_image = None

        image = self.preprocess(Image.open(self.input_image)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.image_features = self.model.encode_image(image)

    def evaluate(self, img, normalization=False):
        img = img.to(self.device)

        # If the image is not available do not contribute to the fitness
        if self.image_features is None:
            return torch.tensor([0])

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
        cosine_similarity = torch.cosine_similarity(self.image_features, image_features, dim=-1).mean()

        return cosine_similarity

