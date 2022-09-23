import clip
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

from render.biggan import BigGAN
from render.renderinterface import RenderingInterface


class CondVectorParameters(torch.nn.Module):
    def __init__(self, ind_numpy, num_latents=15):
        super(CondVectorParameters, self).__init__()
        reshape_array = ind_numpy.reshape(num_latents, -1)
        self.normu = torch.nn.Parameter(torch.tensor(reshape_array).float())
        self.thrsh_lat = torch.tensor(1)
        self.thrsh_cls = torch.tensor(1.9)

    #  def forward(self):
    # return self.ff2(self.ff1(self.latent_code)), torch.softmax(1000*self.ff4(self.ff3(self.cls)), -1)
    #   return self.normu, torch.sigmoid(self.cls)

    # def forward(self):
    #     global CCOUNT
    #     if (CCOUNT < -10):
    #         self.normu,self.cls = copiado(self.normu, self.cls)
    #     if (MAX_CLASSES > 0):
    #         classes = differentiable_topk(self.cls, MAX_CLASSES)
    #         return self.normu, classes
    #     else:
    #         return self.normu#, torch.sigmoid(self.cls)
    def forward(self):
        return self.normu


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        output_size = args.img_size if args.img_size in [128, 256, 512] else 256

        self.model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
        self.model.to(self.device).eval()

        if args.clip is None:
            model, preprocess = clip.load("ViT-B/32", device=args.device)
            self.clip = model
            text_inputs = clip.tokenize([args.target_class]).to(args.device)
            self.text_features = model.encode_text(text_inputs)
        else:
            self.clip = args.clip
            self.text_features = args.text_features

        self.num_latents = len(self.model.config.layers) + 1

        self.genotype_size = (self.num_latents * 256)
        self.real_genotype_size = self.genotype_size

    def chunks(self, array):
        img = np.array(array)
        return np.reshape(img, (self.num_latents, 256))

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self, a, cur_iteration):
        conditional_vector = CondVectorParameters(a, num_latents=self.num_latents).to(self.device)

        lr = 0.07
        num_cuts = 128
        local_search_optimizer = torch.optim.Adam(conditional_vector.parameters(), lr)

        for i in range(5):
            cond_vector = conditional_vector()
            out = self.model(cond_vector, 1)

            p_s = []
            _, channels, sideX, sideY = out.shape
            for ch in range(num_cuts):
                size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
                offsetx = torch.randint(0, sideX - size, ())
                offsety = torch.randint(0, sideX - size, ())
                apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
                p_s.append(torch.nn.functional.interpolate(apper, (224, 224), mode='nearest'))
            # convert_tensor = torchvision.transforms.ToTensor()
            into = torch.cat(p_s, 0)

            into = self.normalize((into + 1) / 2)
            iii = self.clip.encode_image(into)
            cos_similarity = torch.cosine_similarity(self.text_features, iii, dim=-1).mean()

            local_search_optimizer.zero_grad()
            cos_similarity.backward()
            local_search_optimizer.step()

        cond_vector = conditional_vector()
        out = self.model(cond_vector, 1)
        out = TF.to_pil_image(out.squeeze())

        return out


