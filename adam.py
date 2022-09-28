import os

import clip
import torchvision
from PIL.Image import Resampling
from torch import optim
from torchvision.utils import save_image

from utils import CondVectorParameters

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import torchvision.transforms as transforms

cur_iteration = 0


def fitness_classifiers(args, img):
    do_score_reverse = False
    if 'MODEL_REVERSE' in os.environ:
        print("-> predictions reversed")
        do_score_reverse = True

    active_model_keys = sorted(args.active_models.keys())

    # build a table indexed by target_size for all resized image lists
    target_size_table = {}
    for k in active_model_keys:
        model = args.active_models[k]
        target_size = model.get_target_size()
        target_size_table[target_size] = None

    for target_size in target_size_table:
        if target_size is None:
            imr = img
        else:
            imr = img.resize(target_size, resample=Resampling.BILINEAR)
        # target_size_table[target_size].append(tf.keras.utils.img_to_array(imr))
        target_size_table[target_size] = imr

    # convert all lists to np arrays
    # for target_size in target_size_table:
    #     target_size_table[target_size] = np.array(target_size_table[target_size])

    # make all predictions
    full_predictions = []
    fitness_partials = {}
    for k in active_model_keys:
        model = args.active_models[k]
        target_size = model.get_target_size()
        image_preprocessor = model.get_input_preprocessor()

        images = target_size_table[target_size]
        # images = np.copy(target_size_table[target_size])
        if image_preprocessor is not None:
            batch = image_preprocessor(images).unsqueeze(0)
        else:
            batch = images
        preds = model.predict(batch)
        # print("PREDS:", preds.shape, preds)
        if isinstance(preds, dict) and "scores" in preds:
            # print(preds['scores'].shape)
            if len(preds['scores'].shape) == 1:
                worthy = preds['scores']
            elif preds['scores'].shape[1] == 1:
                worthy = preds['scores']
            else:
                worthy = preds['scores'][:, args.imagenet_indexes]
        else:
            worthy = preds[:, args.imagenet_indexes]
            # print("Worthy {}: {}".format(k, np.array(worthy).shape))
            full_predictions.append(worthy)
            fitness_partials[k] = float(worthy)

            # convert predictions to np array
        full_predictions = torch.stack(full_predictions)
        if do_score_reverse:
            print("-> Applying predictions reversed")
            full_predictions = 1.0 - full_predictions

        # top_classes = np.argmax(full_predictions, axis=2).flatten()
        top_classes = torch.argmax(full_predictions, dim=2).flatten()
        # top_class = np.argmax(np.bincount(top_classes))
        top_class = torch.argmax(torch.bincount(top_classes))
        imagenet_index = args.imagenet_indexes[top_class]

        prediction_list = torch.sum(full_predictions, dim=2)

        # extract rewards and merged
        # rewards = np.sum(np.log(prediction_list + 1), axis=0)
        rewards = torch.sum(torch.log(prediction_list + 1), dim=0)
        # merged = np.dstack(prediction_list)[0]
        # merged = torch.dstack(prediction_list)[0]

        return rewards[0]


def fitness_input_image(args, img):
    if not args.clip:
        args.clip_model = "ViT-B/32"
        model, preprocess = clip.load(args.clip_model, device=args.device)
        args.clip = model
        args.preprocess = preprocess

    # Calculate clip similarity
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_t = trans(img).unsqueeze(0).to(args.device)
    image_features = args.clip.encode_image(img_t)
    image_clip_loss = torch.cosine_similarity(args.image_features, image_features, dim=1)
    image_clip_loss *= -1

    return image_clip_loss


def fitness_clip_prompts(args, img):
    if not args.clip:
        args.clip_model = "ViT-B/32"
        model, preprocess = clip.load(args.clip_model, device=args.device)
        args.clip = model
        args.preprocess = preprocess

    # Calculate clip similarity
    p_s = []
    # t_img = TF.to_tensor(img).unsqueeze(0).to(args.device)

    _, channels, sideX, sideY = img.shape
    for ch in range(32):  # TODO - Maybe change here
        size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = img[:, :, offsetx:offsetx + size, offsety:offsety + size]
        p_s.append(torch.nn.functional.interpolate(apper, (224, 224), mode='nearest'))
    # convert_tensor = torchvision.transforms.ToTensor()
    into = torch.cat(p_s, 0).to(args.device)

    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711))
    into = normalize((into + 1) / 2)

    image_features = args.clip.encode_image(into)
    text_clip_loss = torch.cosine_similarity(args.text_features, image_features, dim=-1).mean()
    text_clip_loss *= 100

    return text_clip_loss


def calculate_fitness(args, ind):
    renderer = args.renderer

    # build lists of images at all needed sizes
    # img_array = renderer.chunks(ind)
    img = renderer.render(ind)

    losses = []

    if args.clip_influence < 1.0:
        classifiers_loss = fitness_classifiers(args, img)
        classifiers_loss *= (1.0 - args.clip_influence)
        losses.append(classifiers_loss)

    if args.clip_influence > 0.0:
        text_clip_loss = fitness_clip_prompts(args, img)
        text_clip_loss *= args.clip_influence
        losses.append(text_clip_loss)

    if args.input_image:
        image_clip_loss = fitness_input_image(args, img)
        losses.append(image_clip_loss)

    losses = torch.stack(losses)
    final_loss = torch.sum(losses)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return [final_loss]


def main_adam(args):
    global cur_iteration

    renderer = args.renderer

    individual = renderer.generate_individual()

    if type(individual) is not list:
        individual = torch.nn.Parameter(torch.tensor(individual).float())
        optimizer = optim.Adam([individual], lr=args.lr)
    else:
        optimizer = optim.Adam(individual, lr=args.lr)

    for gen in range(args.n_gens):
        print("Generation:", gen)
        cur_iteration = gen

        loss = calculate_fitness(args, individual)

        optimizer.zero_grad()
        (-loss[0]).backward()
        optimizer.step()

        print(loss[0])

        img = renderer.render(individual)
        img = (img + 1) / 2
        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")
        # img = to_image(img)
        # img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")
