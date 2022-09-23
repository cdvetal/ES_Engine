import clip
import torch
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as TF

from render.biggan import BigGAN


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_size = 256
model = BigGAN.from_pretrained(f'biggan-deep-{output_size}')
model.to(device).eval()

# Load the model
perceptor, preprocess = clip.load('ViT-B/32', device)

text_inputs = clip.tokenize("Darth Vader").to(device)
text_features = perceptor.encode_text(text_inputs)

num_latents = len(model.config.layers) + 1
num_cuts = 128
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

a = torch.rand(num_latents, 256)
conditional_vector = CondVectorParameters(a, num_latents=num_latents).to(device)

optimizer = optim.Adam(conditional_vector.parameters(), lr=0.07)

for i in range(100):
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

    cos_similarity = torch.cosine_similarity(text_features, iii, dim=-1).mean()

    optimizer.zero_grad()
    cos_similarity.backward()
    optimizer.step()

cond_vector = conditional_vector()
out = model(cond_vector, 1)
out = TF.to_pil_image(out)
out.save("out.png")
