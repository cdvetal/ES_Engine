import clip
import torch
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

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
num_cuts = 32
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

a = torch.rand(num_latents, 256)
a.requires_grad = True
conditional_vector = CondVectorParameters(a, num_latents=num_latents).to(device)

optimizer = optim.Adam(conditional_vector.parameters(), lr=0.07)

for i in range(200):
    print(i)

    cond_vector = conditional_vector()
    out = model(cond_vector, 1)

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

    into = normalize((into + 1) / 2)
    iii = perceptor.encode_image(into)  # 128 x 512

    cos_similarity = F.cosine_similarity(text_features, iii, dim=-1).mean()
    cos_similarity = -100 * cos_similarity

    print(cos_similarity)

    optimizer.zero_grad()
    cos_similarity.backward()
    optimizer.step()

cond_vector = conditional_vector()
out = model(cond_vector, 1)
out = TF.to_pil_image(out)
out.save("out.png")
