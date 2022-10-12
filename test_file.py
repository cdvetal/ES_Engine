from PIL import Image
import torchvision.transforms.functional as TF

from ES_Engine.fitnesses import calculate_fitness
from ES_Engine.fitnesses import ClipPrompt

img = Image.open("ES_Engine/dogcat.png").convert('RGB')
img = TF.to_tensor(img).unsqueeze(0)
fits = [ClipPrompt("a painting of superman by Van Gogh")]

fitness = calculate_fitness(fits, img).item()

print(fitness)
