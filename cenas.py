from es_engine import setup_args
from render.vqgan import VQGANRenderer
import torch

args = setup_args()

vq_gan_renderer = VQGANRenderer(args)

t = torch.randn(*vq_gan_renderer.z_shape)

img = vq_gan_renderer.render(t, 128)

print(img)

img.save("image.png")