import os
import pickle
import random
from datetime import datetime
from time import time

import torchvision
from PIL import Image
from PIL.Image import Resampling
import clip
from torch import optim

from utils import save_gen_best, create_save_folder, get_active_models_from_arg, open_class_mapping, \
    get_class_index_list

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
from deap import base
from deap import cma
from deap import creator
from deap import tools
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import argparse
from config import *

from render import *

render_table = {
    "chars": CharsRenderer,
    "pylinhas": PylinhasRenderer,
    "organic": OrganicRenderer,
    "thinorg": ThinOrganicRenderer,
    "pixel": PixelRenderer,
    # "vqgan": VQGANRenderer,
    "clipdraw": ClipDrawRenderer,
    "vdiff": VDiffRenderer,
    "biggan": BigGANRenderer,
}

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

    return image_clip_loss


def fitness_clip_prompts(args, img):
    if not args.clip:
        args.clip_model = "ViT-B/32"
        model, preprocess = clip.load(args.clip_model, device=args.device)
        args.clip = model
        args.preprocess = preprocess

    # Calculate clip similarity
    p_s = []
    t_img = F.to_tensor(img).unsqueeze(0).to(args.device)
    """
    _, channels, sideX, sideY = t_img.shape
    for ch in range(32):  # TODO - Maybe change here
        size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = t_img[:, :, offsetx:offsetx + size, offsety:offsety + size]
        p_s.append(torch.nn.functional.interpolate(apper, (224, 224), mode='nearest'))
    # convert_tensor = torchvision.transforms.ToTensor()
    into = torch.cat(p_s, 0).to(args.device)
    """
    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711))
    resize = torchvision.transforms.Resize((224, 224))
    # into = normalize((into + 1) / 2)
    into = resize(normalize((t_img + 1) / 2))

    image_features = args.clip.encode_image(into)
    text_clip_loss = torch.cosine_similarity(args.text_features, image_features, dim=-1).mean()

    return text_clip_loss


def calculate_fitness(args, ind):
    # build lists of images at all needed sizes
    img_array = args.renderer.chunks(ind)
    img = args.renderer.render(img_array, cur_iteration=cur_iteration)

    losses = []

    if args.clip_influence < 1.0:
        classifiers_loss = fitness_classifiers(args, img)
        losses.append(classifiers_loss)

    if args.clip_influence > 0.0:
        text_clip_loss = fitness_clip_prompts(args, img)
        text_clip_loss *= args.clip_influence
        losses.append(text_clip_loss)

    if args.input_image:
        image_clip_loss = fitness_input_image(args, img)
        losses.append(image_clip_loss)

    losses = torch.stack(losses)
    final_loss = torch.mean(losses)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return [final_loss]


def main_adam(args):
    global cur_iteration

    renderer = args.renderer

    x = torch.rand(renderer.real_genotype_size)
    x.requires_grad = True

    optimizer = optim.Adam([x], lr=0.1)

    for gen in range(args.n_gens):
        print("Generation:", gen)
        cur_iteration = gen

        optimizer.zero_grad()

        loss = calculate_fitness(args, x)

        loss[0].backward()
        optimizer.step()

        img_array = renderer.chunks(x)
        img = renderer.render(img_array, cur_iteration=cur_iteration)
        img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")


def main_cma_es(args):
    global cur_iteration

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", calculate_fitness, args)
    strategy = cma.Strategy(centroid=np.random.normal(args.init_mu, args.init_sigma, args.renderer.real_genotype_size), sigma=args.sigma, lambda_=args.pop_size)  # The genotype size already has the number of lines
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    renderer = args.renderer

    for gen in range(args.n_gens):
        print("Generation:", gen)
        cur_iteration = gen
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            fit = fit[0].cpu().detach().numpy()
            ind.fitness.values = [fit]

        if args.save_all:
            for index, ind in enumerate(population):
                img_array = renderer.chunks(ind)
                img = renderer.render(img_array, cur_iteration=cur_iteration)
                img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_{index}.png")

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        # Update the hall of fame and the statistics with the
        # currently evaluated population
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=gen, **record)

        if args.verbose:
            print(logbook.stream)

        if halloffame is not None:
            save_gen_best(args.save_folder, args.sub_folder, args.experiment_name, [gen, halloffame[0], halloffame[0].fitness.values, "_"])
            img_array = renderer.chunks(halloffame[0])
            img = renderer.render(img_array, cur_iteration=cur_iteration)
            img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")

        if halloffame[0].fitness.values[0] >= args.target_fit:
            print("Reached target fitness.\nExiting")
            break

        if gen % args.checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame, logbook=logbook,
                      np_rndstate=np.random.get_state(), rndstate=random.getstate())
            with open("{}/{}/{}_checkpoint.pkl".format(args.save_folder, args.sub_folder, args.experiment_name), "wb") as cp_file:
                pickle.dump(cp, cp_file)
        # print(time.time() - start)

    print(logbook)


def setup_args():
    parser = argparse.ArgumentParser(description="Evolve to objective")

    parser.add_argument('--evolution-type', default=EVOLUTION_TYPE, help='Specify the type of evolution. (cmaes, adam or hybrid). Default is {}.'.format(EVOLUTION_TYPE))
    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))
    parser.add_argument('--save-folder', default=SAVE_FOLDER, help="Directory to experiment outputs. Default is {}.".format(SAVE_FOLDER))
    parser.add_argument('--n-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--save-all', default=SAVE_ALL, action='store_true', help='Save all Individual images. Default is {}.'.format(SAVE_ALL))
    parser.add_argument('--checkpoint-freq', default=CHECKPOINT_FREQ, type=int, help='Checkpoint save frequency. Default is {}.'.format(CHECKPOINT_FREQ))
    parser.add_argument('--verbose', default=VERBOSE, action='store_true', help='Verbose. Default is {}.'.format(VERBOSE))
    parser.add_argument('--num-lines', default=NUM_LINES, type=int, help="Number of lines. Default is {}".format(NUM_LINES))
    parser.add_argument('--renderer', default=RENDERER, help="Choose the renderer. Default is {}".format(RENDERER))
    parser.add_argument('--img-size', default=IMG_SIZE, type=int, help='Image dimensions during testing. Default is {}.'.format(IMG_SIZE))
    parser.add_argument('--target-class', default=TARGET_CLASS, help='Which target classes to optimize. Default is {}.'.format(TARGET_CLASS))
    parser.add_argument("--networks", default=NETWORKS, help="comma separated list of networks (no spaces). Default is {}.".format(NETWORKS))
    parser.add_argument('--target-fit', default=TARGET_FITNESS, type=float, help='target fitness stopping criteria. Default is {}.'.format(TARGET_FITNESS))
    parser.add_argument('--from-checkpoint', default=FROM_CHECKPOINT, help='Checkpoint file from which you want to continue evolving. Default is {}.'.format(FROM_CHECKPOINT))
    parser.add_argument('--init-mu', default=INIT_MU, type=float, help='Mean value for the initialization of the population. Default is {}.'.format(INIT_MU))
    parser.add_argument('--init-sigma', default=INIT_SIGMA, type=float, help='Standard deviation value for the initialization of the population. Default is {}.'.format(INIT_SIGMA))
    parser.add_argument('--sigma', default=SIGMA, type=float, help='The initial standard deviation of the distribution. Default is {}.'.format(SIGMA))
    parser.add_argument('--clip-influence', default=CLIP_INFLUENCE, type=float, help='The influence CLIP has in the generation (0.0 - 1.0). Default is {}.'.format(CLIP_INFLUENCE))
    parser.add_argument('--clip-model', default=CLIP_MODEL, help='Name of the CLIP model to use. Default is {}. Availables: {}'.format(CLIP_MODEL, clip.available_models()))
    parser.add_argument('--clip-prompts', default=TARGET_CLASS, help='CLIP prompts to use for the generation. Default is the target class')
    parser.add_argument('--input-image', default=None, help='Image to use as input.')

    args = parser.parse_args()

    class_mapping = open_class_mapping()
    if args.target_class is None or args.target_class == "none":
        args.imagenet_indexes = None
    else:
        args.imagenet_indexes = get_class_index_list(class_mapping, args.target_class)

    if args.random_seed:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        # TODO: not do this or maybe there is a tf2 way?
        # tf.compat.v1.set_random_seed(args.random_seed)
        # TODO: Confirm this works
        # tf.random.set_seed(args.random_seed)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", args.device)

    args.renderer = render_table[args.renderer](args)

    args.active_models = get_active_models_from_arg(args.networks)
    args.active_models_quantity = len(args.active_models.keys())

    print("Loaded models:")
    for key, value in args.active_models.items():
        print("- ", key)

    args.clip = None
    if args.clip_influence > 0.0:
        args.clip_influence = min(1.0, max(0.0, args.clip_influence))  # clip value to (0.0 - 1.0)

        if args.clip_model not in clip.available_models():
            args.clip_model = "ViT-B/32"

        print(f"Loading CLIP model: {args.clip_model}")

        model, preprocess = clip.load(args.clip_model, device=args.device)

        print(f"Using \"{args.clip_prompts}\" as prompt to CLIP.")

        # If no clip prompts are given use the target class, else use the provided prompts
        if args.clip_prompts is None:
            text_inputs = clip.tokenize([args.target_class]).to(args.device)
        else:
            text_inputs = clip.tokenize(args.clip_prompts).to(args.device)

        with torch.no_grad():
            args.text_features = model.encode_text(text_inputs)

        args.clip = model
        print("CLIP module loaded.")

        if args.clip_influence == 1.0:
            print("CLIP influence as 1.0. Models removed.")
            args.active_models = {}

    if args.input_image:
        if not os.path.exists(args.input_image):
            print("Image file does not exist. Ignoring..")
            args.input_image = None
        else:
            image = args.preprocess(Image.open(args.input_image)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                args.image_features = args.clip.encode_image(image)

    if args.from_checkpoint:
        args.experiment_name = args.from_checkpoint.replace("_checkpoint.pkl", "")
        # save_folder = f"experiments/{experiment_name}"
        # CHECKPOINT = f"{save_folder}/{CHECKPOINT}"
        # save_folder = "{}/{}".format(save_folder, experiment_name)
        args.sub_folder = "from_checkpoint"
        save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)
        args.checkpoint = "{}/{}".format(save_folder, args.from_checkpoint)
    else:
        args.experiment_name = f"{args.renderer}_L{args.num_lines}_{args.target_class}_CLIP{args.clip_influence}_{args.random_seed if args.random_seed else datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        args.sub_folder = f"{args.experiment_name}_{args.n_gens}_{args.pop_size}"
        save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)
        args.checkpoint = "{}/{}".format(save_folder, args.from_checkpoint)

    return args


if __name__ == "__main__":
    # Get time of start of the program
    start_time_total = time()
    # Get arguments
    args = setup_args()
    # Get time of start of the evolution
    start_time_evo = time()
    # Main program
    # main_adam(args)
    main_cma_es(args)
    # Get end time
    end_time = time()

    total_time = (end_time - start_time_total)
    evo_time = (end_time - start_time_evo)

    print("-" * 20)
    print("Evolution elapsed time: {:.3f}".format(evo_time))
    print("Total elapsed time: {:.3f}".format(total_time))
    print("-" * 20)
