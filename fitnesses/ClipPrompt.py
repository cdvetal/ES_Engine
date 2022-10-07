import clip
import torch
import torchvision

from fitnesses.fitness_interface import FitnessInterface


class ClipPrompt(FitnessInterface):
    def __init__(self, prompts, model=None, preprocess=None, clip_model="ViT-B/32"):
        super(ClipPrompt, self).__init__()

        self.clip_model = "ViT-B/32"

        if model is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

        self.prompts = prompts

        text_inputs = clip.tokenize(self.prompts).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)

    def evaluate(self, img):
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

        normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                     (0.26862954, 0.26130258, 0.27577711))
        into = normalize((into + 1) / 2)

        image_features = self.model.encode_image(into)
        text_clip_loss = torch.cosine_similarity(self.text_features, image_features, dim=-1).mean()
        text_clip_loss *= 1

        return text_clip_loss

