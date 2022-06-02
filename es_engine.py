import os
import random
from datetime import datetime

from PIL import Image
import tensorflow as tf
import torch
from deap import gp, creator, base, tools, algorithms, cma

from utils import save_gen_best, create_save_folder, get_active_models_from_arg, open_class_mapping, \
    get_class_index_list

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
"""
from deap import base
from deap import cma
from deap import creator
from deap import tools
import torch
"""
import argparse
from config import *


def clip_fitness(individual):
    # global COUNT_IND, COUNT_GENERATION
    ind_array = np.array(individual)
    conditional_vector = big_sleep_cma_es.CondVectorParameters(ind_array, batch_size=BATCH_SIZE)  #
    result = big_sleep_cma_es.evaluate_with_local_search(conditional_vector, 10)
    # big_sleep.checkin_with_cond_vectors(result, conditional_vector, individual=COUNT_IND, itt=COUNT_GENERATION)
    # COUNT_IND += 1
    # print("Lamack", LAMARCK)
    if LAMARCK:
        individual[:] = conditional_vector().cpu().detach().numpy().flatten()
    return float(result[2].float().cpu()) * -1,
    # return (float(result[0].float().cpu()) * -1) / 10000+ (float(result[1].float().cpu()) * -1)/10000  + (float(result[2].float().cpu()) * -1)*1,


"""
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", clip_fitness)
"""


def generate_individual_with_embeddings(batch_size):
    latent = torch.nn.Parameter(torch.zeros(batch_size, 128).normal_(std=1).float().cuda())
    params_other = torch.zeros(batch_size, 1000).normal_(-3.9, .3).cuda()
    classes = torch.sigmoid(torch.nn.Parameter(params_other))
    embed = big_sleep_cma_es.model.embeddings(classes)
    cond_vector = torch.cat((latent, embed), dim=1)
    ind = cond_vector.cpu().detach().numpy().flatten()
    # cond_vector = big_sleep_cma_es.CondVectorParameters(ind, batch_size=BATCH_SIZE)
    # big_sleep_cma_es.save_individual_cond_vector(cond_vector, f"PONTO_INICIAL.png")
    return ind


def evaluate_fitness(individual, args):
    # TODO - Generate the phenotype
    # TODO - Evaluate phenotype
    # TODO - Calculate fitness
    ind_array = np.array(individual)

def keras_fitness(ind, img_width, img_height):
    do_score_reverse = False
    if 'MODEL_REVERSE' in os.environ:
        print("-> predictions reversed")
        do_score_reverse = True


    active_model_keys = sorted(ACTIVE_MODELS.keys())

    # build a table indexed by target_size for all resized image lists
    target_size_table = {}
    for k in active_model_keys:
        model = ACTIVE_MODELS[k]
        target_size = model.get_target_size()
        target_size_table[target_size] = []

    # build lists of images at all needed sizes
    img_array = chunks(ind)
    img = RENDER_IMAGE(img_array, size=img_height)
    for target_size in target_size_table:
        if target_size is None:
            imr = img
        else:
            imr = img.resize(target_size, resample=Image.BILINEAR)
        target_size_table[target_size].append(image.img_to_array(imr))

    # # convert all lists to np arrays
    for target_size in target_size_table:
        target_size_table[target_size] = np.array(target_size_table[target_size])

    # make all predictions
    full_predictions = []
    fitness_partials = {}
    for k in active_model_keys:
        model = ACTIVE_MODELS[k]
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
                worthy = preds['scores'][:, IMAGENET_INDEXES]
        else:
            worthy = preds[:, IMAGENET_INDEXES]
        # print("Worthy {}: {}".format(k, np.array(worthy).shape))
        full_predictions.append(worthy)
        fitness_partials[k] = float(worthy)

    # convert predictions to np array
    full_predictions = np.array(full_predictions)
    if do_score_reverse:
        print("-> Applying predictions reversed")
        full_predictions = 1.0 - full_predictions
    top_classes = np.argmax(full_predictions, axis=2).flatten()
    top_class = np.argmax(np.bincount(top_classes))
    imagenet_index = IMAGENET_INDEXES[top_class]

    prediction_list = np.sum(full_predictions, axis=2)

    # extract rewards and merged
    rewards = np.sum(np.log(prediction_list + 1), axis=0)
    merged = np.dstack(prediction_list)[0]
    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    return [(rewards[0],), fitness_partials]


def main(args):
    print(args)

    # The cma module uses the np random number generator
    save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", clip_fitness)
    strategy = cma.Strategy(centroid=generate_individual_with_embeddings(), sigma=0.2, lambda_=args.pop_size)
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

    """
    for gen in range(N_GENS):
        COUNT_GENERATION = gen
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        COUNT_IND = 0
        COUNT_GENERATION = gen
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if SAVE_ALL:
            for index, ind in enumerate(population):
                cond_vector = big_sleep_cma_es.CondVectorParameters(np.array(ind), batch_size=BATCH_SIZE)
                big_sleep_cma_es.save_individual_cond_vector(cond_vector,
                                                             f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_{index}.png")

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
            save_gen_best(save_folder, sub_folder, experiment_name,
                                      [gen, halloffame[0], halloffame[0].fitness.values, "_"])
            cond_vector = big_sleep_cma_es.CondVectorParameters(np.array(halloffame[0]), batch_size=BATCH_SIZE)
            big_sleep_cma_es.save_individual_cond_vector(cond_vector,
                                                         f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_best.png")
    """


def setup_args():
    parser = argparse.ArgumentParser(description="Evolve to objective")

    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))
    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is {}.".format(SAVE_FOLDER))
    parser.add_argument('--n-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--save-all', default=False, action='store_true', help='Save all Individual images. Default is False.')
    parser.add_argument('--lamarck', default=False, action='store_true', help='Lamarckian evolution. Default is False.')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose. Default is False.')
    parser.add_argument('--num-lines', default=NUM_LINES, type=int, help="Number of lines. Default is {}".format(NUM_LINES))
    parser.add_argument('--num-cols', default=NUM_COLS, type=int, help="Number of columns. Default is {}".format(NUM_COLS))
    parser.add_argument('--renderer', default=RENDERER, help="Choose the renderer. Default is {}".format(RENDERER))
    parser.add_argument('--img-size', default=IMG_SIZE[0], type=int, help='Image dimensions during testing. Default is {}.'.format(IMG_SIZE[0]))
    parser.add_argument('--target-class', default=TARGET_CLASS, help='which target classes to optimize. Default is {}.'.format(TARGET_CLASS))
    parser.add_argument("--networks", default=NETWORKS, help="comma separated list of networks (no spaces). Default is {}.".format(NETWORKS))
    parser.add_argument('--target-fit', default=TARGET_FITNESS, type=float, help='target fitness stopping criteria. Default is {}.'.format(TARGET_FITNESS))
    parser.add_argument('--checkpoint-freq', default=CHECKPOINT_FREQ, help='Checkpoint backup frequency. Default is every {} generations.'.format(CHECKPOINT_FREQ))
    parser.add_argument('--from-checkpoint', default=FROM_CHECKPOINT, help='Checkpoint file from which you want to continue evolving. Default is {}.'.format(FROM_CHECKPOINT))
    parser.add_argument('--tournament-size', default=TOURNAMENT_SIZE, type=int, help='Tournament size for selection operator. Default is {}.'.format(TOURNAMENT_SIZE))
    parser.add_argument('--cxpb', default=CXPB, type=float, help='Crossover probability. Default is {}.'.format(CXPB))
    parser.add_argument('--mutpb', default=MUTPB, type=float, help='Mutation probability. Default is {}.'.format(MUTPB))
    parser.add_argument('--mut-mu', default=MUT_MU, type=float, help='Mean or python:sequence of means for the gaussian addition mutation. Default is {}.'.format(MUT_MU))
    parser.add_argument('--mut-sigma', default=MUT_SIGMA, type=float, help='Standard deviation or python:sequence of standard deviations for the gaussian addition mutation. Default is {}.'.format(MUT_SIGMA))
    parser.add_argument('--mut-indpb', default=MUT_INDPB, type=float, help='Independent probability for each attribute to be mutated. Default is {}.'.format(MUT_INDPB))

    args = parser.parse_args()

    args.sub_folder = f"{args.n_gens}_{args.pop_size}"

    print(args)

    if args.from_checkpoint:
        experiment_name = args.from_checkpoint.replace("_checkpoint.pkl", "")
        # save_folder = f"experiments/{experiment_name}"
        # CHECKPOINT = f"{save_folder}/{CHECKPOINT}"
        # save_folder = "{}/{}".format(save_folder, experiment_name)
        sub_folder = "from_checkpoint"
        save_folder, sub_folder = create_save_folder(args.save_folder, sub_folder)
        args.checkpoint = "{}/{}".format(save_folder, args.from_checkpoint)
    else:
        experiment_name = f"pylinhas_{args.renderer}_L{args.num_lines}_C{args.num_cols}_{args.target_class}_{args.random_seed if args.random_seed else datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        sub_folder = f"{experiment_name}_{args.n_gens}_{args.pop_size}"
        save_folder, sub_folder = create_save_folder(args.save_folder, sub_folder)

    ACTIVE_MODELS = get_active_models_from_arg(args.networks)
    # ACTIVE_MODELS_QUANTITY = len(ACTIVE_MODELS.keys())

    for key, value in ACTIVE_MODELS.items():
        pass
        print("Key", key)
        print("Model", value.model)

    class_mapping = open_class_mapping()
    if TARGET_CLASS is None or TARGET_CLASS == "none":
        args.imagenet_indexes = None
    else:
        args.imagenet_indexes = get_class_index_list(class_mapping, TARGET_CLASS)

    if args.random_seed:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        # TODO: not do this or maybe there is a tf2 way?
        # tf.compat.v1.set_random_seed(args.random_seed)
        # TODO: Confirm this works
        tf.random.set_seed(args.random_seed)

    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
