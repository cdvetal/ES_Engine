from datetime import datetime
import random
import tensorflow as tf
import tensorflow.keras as keras
import time
import warnings
import numpy as np
import os
import pickle
from extra_tools import save_gen_best, create_save_folder, save_experimental_setup, save_logbook_as_csv
from PIL import Image
from deap import gp, creator, base, tools, algorithms
from tensorflow.keras.preprocessing import image
from classloader import ScoringInterface, load_image_function
from utils import open_class_mapping, get_class_index_list, get_active_models_from_arg
import argparse
# from profiler import profile
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["SM_FRAMEWORK"] = "tf.keras"

IMG_SIZE = IMG_WIDTH, IMG_HEIGHT = (512, 512)  # ATTENTION!!!! Only square images now please.

NUM_LINES = 17
NUM_COLS = 8

RENDER_IMAGE = None


POP_SIZE = 10
N_GENS = 2
ELITISM = 1
CXPB = 0.9
MUTPB = 0.1
# Parameters for Gaussian Mutation
MUT_MU = 0
MUT_SIGMA = 1
MUT_INDPB = 0.2

TOURNAMENT_SIZE = 5

TARGET_FITNESS = 0.999,
RANDOM_SEED = None

CHECKPOINT_FREQ = 10
CHECKPOINT = None  # None or "Experiment_name.pkl""

NETWORKS = "mobilenet,mobilenetv2"
ACTIVE_MODELS = None
ACTIVE_MODELS_QUANTITY = None


TARGET_CLASS = "goldfish"

IMAGENET_INDEXES = None

SAVE_ALL = False

# ---------------------------
# ---------------------------


def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator


def chunks(array):
    img = np.array(array)
    return np.reshape(img, (NUM_LINES, NUM_COLS))


def cxTwoPointCopy(ind1, ind2):  # funcao de crossover
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


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
            if(len(preds['scores'].shape) == 1):
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


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__, experiment_name=None, save_folder=None, sub_folder=None, checkpoint=None, render_image=None):

    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        np.random.set_state(cp["np_rndstate"])
        random.setstate(cp["rndstate"])
        # TODO: Confirm if this is ok.
        tf.compat.v1.set_random_seed(cp["rndstate"])

    else:
        start_gen = 0
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        logbook.chapters["fitness"].header = "max", "avg", "min", "std"
        for k in stats.fields:
            logbook.chapters[k].header = "max", "avg", "min", "std"

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        eval_results = toolbox.map(toolbox.evaluate, invalid_ind)


        for ind, eval_result in zip(invalid_ind, eval_results):
            ind.fitness.values = eval_result[0]
            ind.fitness_partials = eval_result[1]
        
        
        if SAVE_ALL:
            for index, ind in enumerate(population):
                img_array = chunks(ind)
                img = render_image(img_array, size=IMG_HEIGHT)
                img.save(f"{save_folder}/{sub_folder}/{experiment_name}_0_{index}.png")
        
        
        if halloffame is not None:
            halloffame.update(population)
            save_gen_best(save_folder, sub_folder, experiment_name, [0, str(halloffame[0]), halloffame[0].fitness.values, "_"])
            img_array = chunks(halloffame[0])
            img = render_image(img_array, size=IMG_HEIGHT)
            img.save(f"{save_folder}/{sub_folder}/{experiment_name}_0_best.png")

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        save_logbook_as_csv(save_folder, sub_folder, experiment_name, logbook)
        if verbose:
            print(logbook.stream)

    # Begin the generational process
    gen = start_gen


    while halloffame[0].fitness.values < TARGET_FITNESS and gen < ngen:
        # while gen < ngen:  # for JPG size maximization example test
        # start = time.time()
        gen += 1
        # Select the next generation individuals
        selected = toolbox.select(population, POP_SIZE)
        offspring = [toolbox.clone(ind) for ind in selected]

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        eval_results = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, eval_result in zip(invalid_ind, eval_results):
            ind.fitness.values = eval_result[0]
            ind.fitness_partials = eval_result[1]

        #if SAVE_ALL:
        #    for 
        #    dex, ind in enumerate(offspring):
        #        ##img_array = chunks(ind, NUM_COLS)
        #        img = render_image(img_array, size=IMG_HEIGHT)
        #        img.save(f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_{index}.png")

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            save_gen_best(save_folder, sub_folder, experiment_name, [gen, str(halloffame[0]), halloffame[0].fitness.values, "_"])
            img_array = chunks(halloffame[0])
            img = render_image(img_array, size=IMG_HEIGHT)
            img.save(f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_best.png")

        # Replace the current population by the offspring
        population[:] = tools.selBest(population, ELITISM) + offspring[ELITISM:]

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        save_logbook_as_csv(save_folder, sub_folder, experiment_name, logbook)
        if verbose:
            print(logbook.stream)

        if gen % CHECKPOINT_FREQ == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame, logbook=logbook, np_rndstate=np.random.get_state(), rndstate=random.getstate())
            with open("{}/{}/{}_checkpoint.pkl".format(save_folder, sub_folder, experiment_name), "wb") as cp_file:
                pickle.dump(cp, cp_file)
        # print(time.time() - start)
    return population, logbook, halloffame


# @profile
def main():
    global POP_SIZE, IMG_SIZE, IMG_WIDTH, IMG_HEIGHT, N_GENS, TARGET_FITNESS, RANDOM_SEED, NETWORKS, TARGET_CLASS, IMAGENET_INDEXES, CHECKPOINT, CHECKPOINT_FREQ, ACTIVE_MODELS, SAVE_ALL, NUM_LINES, NUM_COLS, RENDER_IMAGE, TOURNAMENT_SIZE, CXPB, MUTPB, MUT_MU, MUT_SIGMA, MUT_INDPB

    parser = argparse.ArgumentParser(description="evolve to objective")
    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is 'experiments'.")
    parser.add_argument('--num-lines', default=NUM_LINES, type=int)
    parser.add_argument('--num-cols', default=NUM_COLS, type=int)
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--max-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))
    parser.add_argument('--renderer', default="pylinhas", help="Choose the renderer. Default is 'pylinhas'")
    parser.add_argument('--img-size', default=IMG_SIZE[0], type=int, help='Image dimensions during testing. Default is {}.'.format(IMG_SIZE[0]))
    parser.add_argument('--target-class', default=TARGET_CLASS, help='which target classes to optimize. Default is {}.'.format(TARGET_CLASS))
    parser.add_argument("--networks", default=NETWORKS, help="comma separated list of networks (no spaces). Default is {}.".format(NETWORKS))
    parser.add_argument('--target-fit', default=TARGET_FITNESS, type=float, help='target fitness stopping criteria. Default is {}.'.format(TARGET_FITNESS))
    parser.add_argument('--checkpoint-freq', default=CHECKPOINT_FREQ, help='Checkpoint backup frequency. Default is every {} generations.'.format(CHECKPOINT_FREQ))
    parser.add_argument('--from-checkpoint', default=CHECKPOINT, help='Checkpoint file from which you want to continue evolving. Default is {}.'.format(CHECKPOINT))
    parser.add_argument('--save-all', default=SAVE_ALL, action='store_true', help='Save all Individual images. Default is {}.'.format(SAVE_ALL))
    parser.add_argument('--tournament-size', default=TOURNAMENT_SIZE, type=int, help='Tournament size for selection operator. Default is {}.'.format(TOURNAMENT_SIZE))
    parser.add_argument('--cxpb', default=CXPB, type=float, help='Crossover probability. Default is {}.'.format(CXPB))
    parser.add_argument('--mutpb', default=MUTPB, type=float, help='Mutation probability. Default is {}.'.format(MUTPB))
    parser.add_argument('--mut-mu', default=MUT_MU, type=float, help='Mean or python:sequence of means for the gaussian addition mutation. Default is {}.'.format(MUT_MU))
    parser.add_argument('--mut-sigma', default=MUT_SIGMA, type=float, help='Standard deviation or python:sequence of standard deviations for the gaussian addition mutation. Default is {}.'.format(MUT_SIGMA))
    parser.add_argument('--mut-indpb', default=MUT_INDPB, type=float, help='Independent probability for each attribute to be mutated. Default is {}.'.format(MUT_INDPB))

    args = parser.parse_args()

    NUM_COLS = int(args.num_cols)
    NUM_LINES = int(args.num_lines)
    RENDER_IMAGE = load_image_function(args.renderer + '.render')

    save_folder = args.save_folder
    POP_SIZE = int(args.pop_size)
    IMG_SIZE = IMG_WIDTH, IMG_HEIGHT = (int(args.img_size), int(args.img_size))
    N_GENS = int(args.max_gens)
    TARGET_FITNESS = args.target_fit,
    RANDOM_SEED = args.random_seed
    NETWORKS = args.networks
    TARGET_CLASS = args.target_class
    CHECKPOINT = args.from_checkpoint
    CHECKPOINT_FREQ = int(args.checkpoint_freq)
    SAVE_ALL = args.save_all
    TOURNAMENT_SIZE = args.tournament_size
    CXPB = args.cxpb
    MUTPB = args.mutpb
    # Parameters for Gaussian Mutation
    MUT_MU = args.mut_mu
    MUT_SIGMA = args.mut_sigma
    MUT_INDPB = args.mut_indpb

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, fitness_partials=dict)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.02, 0.98)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_COLS*NUM_LINES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutGaussian, mu=MUT_MU, sigma=MUT_SIGMA, indpb=MUT_INDPB)

    toolbox.decorate("mate", checkBounds(0.02, 0.98))
    toolbox.decorate("mutate", checkBounds(0.02, 0.98))

    toolbox.register("clip", np.clip, a_min=0.02, a_max=0.98)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    toolbox.register("evaluate", keras_fitness, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    if CHECKPOINT:
        experiment_name = CHECKPOINT.replace("_checkpoint.pkl", "")
        # save_folder = f"experiments/{experiment_name}"
        # CHECKPOINT = f"{save_folder}/{CHECKPOINT}"
        # save_folder = "{}/{}".format(save_folder, experiment_name)
        sub_folder = "from_checkpoint"
        save_folder, sub_folder = create_save_folder(save_folder, sub_folder)
        CHECKPOINT = "{}/{}".format(save_folder, CHECKPOINT)
    else:
        experiment_name = f"pylinhas_{args.renderer}_L{NUM_LINES}_C{NUM_COLS}_{TARGET_CLASS}_{RANDOM_SEED if RANDOM_SEED else datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        sub_folder = f"{experiment_name}_{N_GENS}_{POP_SIZE}"
        save_folder, sub_folder = create_save_folder(save_folder, sub_folder)

    ACTIVE_MODELS = get_active_models_from_arg(NETWORKS)
    ACTIVE_MODELS_QUANTITY = len(ACTIVE_MODELS.keys())

    class_mapping = open_class_mapping()
    if TARGET_CLASS is None or TARGET_CLASS == "none":
        IMAGENET_INDEXES = None
    else:
        IMAGENET_INDEXES = get_class_index_list(class_mapping, TARGET_CLASS)

    if args.random_seed:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        # TODO: not do this or maybe there is a tf2 way?
        tf.compat.v1.set_random_seed(args.random_seed)

    pop = toolbox.population(n=POP_SIZE)
    hof = hof = tools.HallOfFame(1, similar=np.array_equal)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats_mobilenet = tools.Statistics(lambda ind: ind.fitness_partials["mobilenetv2"])
    # stats_mobilenetv2 = tools.Statistics(lambda ind: ind.fitness_partials["mobilenetv2"])
    # mstats = tools.MultiStatistics(fitness=stats_fit, mobilenet=stats_mobilenet, mobilenetv2=stats_mobilenetv2)

    def create_stats_from_active_models(k):
        return tools.Statistics(lambda ind: ind.fitness_partials[k])

    stats_from_active_networks = {k: create_stats_from_active_models(k) for k in ACTIVE_MODELS.keys()}
    mstats = tools.MultiStatistics(fitness=stats_fit, **stats_from_active_networks)

    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    save_experimental_setup(save_folder, sub_folder, experiment_name, args)

    pop, logbook, hof = eaSimple(pop, toolbox, CXPB, MUTPB, N_GENS, mstats, halloffame=hof, verbose=False, experiment_name=experiment_name, save_folder=save_folder, sub_folder=sub_folder, checkpoint=CHECKPOINT, render_image=RENDER_IMAGE)  # , checkpoint=".pkl"


if __name__ == "__main__":
    main()
