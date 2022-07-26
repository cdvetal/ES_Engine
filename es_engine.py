import os
import random
from datetime import datetime

from PIL import Image
from PIL.Image import Resampling
import tensorflow as tf
import clip

from utils import save_gen_best, create_save_folder, get_active_models_from_arg, open_class_mapping, \
    get_class_index_list

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from deap import base
from deap import cma
from deap import creator
from deap import tools
import torch
import torchvision.transforms as transforms
import argparse
from config import *

from render.chars import CharsRenderer
from render.pylinhas import PylinhasRenderer
from render.organic import OrganicRenderer
from render.thinorg import ThinOrganicRenderer
from render.vqgan import VQGANRenderer

render_table = {
    "chars": CharsRenderer,
    "pylinhas": PylinhasRenderer,
    "organic": OrganicRenderer,
    "thinorg": ThinOrganicRenderer,
    "vqgan": VQGANRenderer
}

# TODO - Use GPU if available
# TODO - Experiment with VQGAN


def keras_fitness(args, ind):
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
        target_size_table[target_size] = []

    # build lists of images at all needed sizes
    img_array = args.renderer.chunks(ind)
    img = args.renderer.render(img_array, img_size=args.img_size)

    for target_size in target_size_table:
        if target_size is None:
            imr = img
        else:
            imr = img.resize(target_size, resample=Resampling.BILINEAR)
        target_size_table[target_size].append(tf.keras.utils.img_to_array(imr))

    # convert all lists to np arrays
    for target_size in target_size_table:
        target_size_table[target_size] = np.array(target_size_table[target_size])

    # make all predictions
    full_predictions = []
    fitness_partials = {}
    for k in active_model_keys:
        model = args.active_models[k]
        target_size = model.get_target_size()
        image_preprocessor = model.get_input_preprocessor()

        # images = target_size_table[target_size]
        images = np.copy(target_size_table[target_size])
        if image_preprocessor is not None:
            batch = image_preprocessor(images)
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
    full_predictions = np.array(full_predictions)
    if do_score_reverse:
        print("-> Applying predictions reversed")
        full_predictions = 1.0 - full_predictions

    if np.size(full_predictions):  # Check if there are prediction, only happens when no network is specified
        top_classes = np.argmax(full_predictions, axis=2).flatten()
        top_class = np.argmax(np.bincount(top_classes))
        imagenet_index = args.imagenet_indexes[top_class]

        prediction_list = np.sum(full_predictions, axis=2)

        # extract rewards and merged
        rewards = np.sum(np.log(prediction_list + 1), axis=0)
        merged = np.dstack(prediction_list)[0]
    else:
        rewards = [0.0]

    # Calculate clip similarity
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_t = trans(img).unsqueeze(0)
    image_features = args.clip.encode_image(img_t)
    loss = torch.cosine_similarity(args.text_features, image_features, dim=1).item()

    final_value = ((loss * args.clip_influence) + (rewards[0] * (1.0 - args.clip_influence)))
    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return [final_value]


def main(args):
    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", keras_fitness, args)
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
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if args.save_all:
            for index, ind in enumerate(population):
                img_array = renderer.chunks(ind)
                img = renderer.render(img_array, img_size=args.img_size)
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
            img = renderer.render(img_array, img_size=args.img_size)
            img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")

    print(logbook)


def setup_args():
    parser = argparse.ArgumentParser(description="Evolve to objective")

    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))
    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is {}.".format(SAVE_FOLDER))
    parser.add_argument('--n-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--save-all', default=SAVE_ALL, action='store_true', help='Save all Individual images. Default is {}.'.format(SAVE_ALL))
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

    args = parser.parse_args()

    args.sub_folder = f"{args.n_gens}_{args.pop_size}"

    if args.from_checkpoint:
        args.experiment_name = args.from_checkpoint.replace("_checkpoint.pkl", "")
        # save_folder = f"experiments/{experiment_name}"
        # CHECKPOINT = f"{save_folder}/{CHECKPOINT}"
        # save_folder = "{}/{}".format(save_folder, experiment_name)
        args.sub_folder = "from_checkpoint"
        save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)
        args.checkpoint = "{}/{}".format(save_folder, args.from_checkpoint)
    else:
        args.experiment_name = f"{args.renderer}_L{args.num_lines}_{args.target_class}_{args.random_seed if args.random_seed else datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        args.sub_folder = f"{args.experiment_name}_{args.n_gens}_{args.pop_size}"
        save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)
        args.checkpoint = "{}/{}".format(save_folder, args.from_checkpoint)

    args.active_models = get_active_models_from_arg(args.networks)
    args.active_models_quantity = len(args.active_models.keys())

    print("Loaded models:")
    for key, value in args.active_models.items():
        print("- ", key)

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
        tf.random.set_seed(args.random_seed)

    args.renderer = render_table[args.renderer](args)

    if args.clip_influence > 0.0:
        args.clip_influence = min(1.0, max(0.0, args.clip_influence))  # clip value to (0.0 - 1.0)
        # If no models to guide the evolution use clip at fullest
        if args.active_models_quantity == 0:
            args.clip_influence = 1.0
            print("No active model, CLIP influence changed to 1.0")

        model, preprocess = clip.load(args.clip_model, "cpu")

        # If no clip prompts are given use the target class, else use the provided prompts
        if args.clip_prompts is None:
            text_inputs = clip.tokenize([args.target_class])
        else:
            text_inputs = clip.tokenize(args.clip_prompts)

        args.text_features = model.encode_text(text_inputs)
        args.clip = model
        print("CLIP module loaded.")

    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
