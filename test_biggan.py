import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from render.biggan import BigGAN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_size = 256
model = BigGAN.from_pretrained(f'biggan-deep-{256}')
model.to(device).eval()

num_latents = len(model.config.layers) + 1


z_dim = 128
latent = torch.nn.Parameter(torch.zeros(num_latents, z_dim).normal_(std=1).float().cuda())
params_other = torch.zeros(num_latents, 1000).normal_(-3.9, .3).cuda()
classes = torch.sigmoid(torch.nn.Parameter(params_other))
embed = model.embeddings(classes)
cond_vector = torch.cat((latent, embed), dim=1)
# ind = cond_vector.cpu().detach().numpy().flatten()
print(cond_vector.shape)

out = model(cond_vector, 1)
out = (out + 1) / 2
save_image(out, "out.png")
