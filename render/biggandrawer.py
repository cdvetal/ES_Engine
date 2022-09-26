import torch
import torchvision.transforms.functional as TF

from render.biggan import BigGAN
from render.renderinterface import RenderingInterface


class BigGANRenderer(RenderingInterface):
    def __init__(self, args):
        super(BigGANRenderer, self).__init__(args)

        self.device = args.device

        output_size = args.img_size if args.img_size in [128, 256, 512] else 256

        self.model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
        self.model.to(self.device).eval()

        self.num_latents = len(self.model.config.layers) + 1

        self.genotype_size = (self.num_latents * 256)
        self.real_genotype_size = self.genotype_size

    def generate_individual(self):
        z_dim = 128
        latent = torch.nn.Parameter(torch.zeros(self.num_latents, z_dim).normal_(std=1).float().cuda())
        params_other = torch.zeros(self.num_latents, 1000).normal_(-3.9, .3).cuda()
        classes = torch.sigmoid(torch.nn.Parameter(params_other))
        embed = self.model.embeddings(classes)
        cond_vector = torch.cat((latent, embed), dim=1)
        ind = cond_vector.cpu().detach().numpy().flatten()
        return ind

    def chunks(self, array):
        array = torch.tensor(array, dtype=torch.float)
        return array.view(self.num_latents, 256)

    def __str__(self):
        return "biggan"

    # input: array of real vectors, length 8, each component normalized 0-1
    def render(self, a, cur_iteration):
        a = a.float().to(self.device)

        out = self.model(a, 1)
        out = TF.to_pil_image(out.squeeze())

        return out


