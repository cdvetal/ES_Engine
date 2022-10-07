from pathlib import Path

import torch

model_path = Path("models/ava_vit_b_16_linear.pth")

layer_weights = torch.load(model_path)
