from PIL import Image
import numpy as np

from utils import get_active_models_from_arg

networks = "vgg16,vgg19,mobilenet"
ACTIVE_MODELS = get_active_models_from_arg(networks)
# ACTIVE_MODELS_QUANTITY = len(ACTIVE_MODELS.keys())

img = Image.open('dogcat.png')
img = np.asarray(img)

for key, value in ACTIVE_MODELS.items():
    print("Key", key)
    print("Model", value.model)
    processor = value.get_input_preprocessor()
    img_p = processor(img)
    print(img_p.shape)
