import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["pink flamingo"]).to(device)
with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)


def call_clip(image_nevar):    

    image = preprocess(image_nevar).unsqueeze(0).to(device)
   

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        # text_features = model.encode_text(text)
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return similarity.cpu().numpy()


# # Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)

# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# values, indices = similarity[0].topk(5)

# # Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")