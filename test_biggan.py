import clip
import torch
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as TF

from render.biggan import BigGAN
from utils import CondVectorParameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_size = 256
model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
model.to(device).eval()

# Load the model
perceptor, preprocess = clip.load('ViT-B/32', device)

text_inputs = clip.tokenize("Darth Vader").to(device)

with torch.no_grad():
    text_features = perceptor.encode_text(text_inputs)

num_latents = len(model.config.layers) + 1

augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

a = torch.rand(num_latents, 256)
a.requires_grad = True
conditional_vector = CondVectorParameters(a).to(device)

optimizer = optim.Adam(conditional_vector.parameters(), lr=0.1)

for i in range(200):
    print(i)

    optimizer.zero_grad()

    cond_vector = conditional_vector()
    out = model(cond_vector, 1)

    loss = 0
    NUM_AUGS = 4
    img_augs = []
    for n in range(NUM_AUGS):
        img_augs.append(augment_trans(out))
    im_batch = torch.cat(img_augs)
    image_features = model.encode_image(im_batch)
    for n in range(NUM_AUGS):
        loss -= torch.cosine_similarity(text_features, image_features[n:n + 1], dim=1)

    loss.backward()
    optimizer.step()

cond_vector = conditional_vector()
out = model(cond_vector, 1)
out = TF.to_pil_image(out.squeeze())
out.save("out.png")
