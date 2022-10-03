from torchvision.utils import save_image

from main import setup_args
from render import VDiffRenderer

args = setup_args()

renderer = VDiffRenderer(args)

i = renderer.generate_individual()
print(i.shape)

cena = renderer.to_adam(i)
print(cena[0].shape)

img = renderer.render(cena)
print(img)

img = (img + 1) / 2

save_image(img, "out.png")
