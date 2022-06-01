# from profiler import profile
import argparse
from utils import open_class_mapping, get_class_index_list, get_active_models_from_arg
from classloader import ScoringInterface
from tensorflow.keras.preprocessing import image
from deap import gp, creator, base, tools, algorithms
from PIL import Image
from extra_tools import save_gen_best, create_save_folder, save_img, save_logbook_as_csv, image_generation, IMG_SIZE, IMG_WIDTH, IMG_HEIGHT
import pickle
import math
import operator
import os
import numpy as np
import warnings
import time
import tensorflow as tf
import tensorflow.keras as keras
import random
from datetime import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)
import time
#import cliptest
#from tensorgp.engine import *


from tensorgp.tensorgp_v2 import *
import random


######################## PYNEVAR territory
POP_SIZE = 100
N_GENS = 50
ELITISM = 1
CXPB = 0.0
MUTPB = 0.6

TOURNAMENT_SIZE = 5

TARGET_FITNESS = 0.999,
RANDOM_SEED = None

CHECKPOINT_FREQ = 10
CHECKPOINT = None  # None or "Experiment_name.pkl""

NETWORKS = "vgg19"
do_bgr = True if NETWORKS is "vgg19" else False
ACTIVE_MODELS = None
ACTIVE_MODELS_QUANTITY = None


TARGET_CLASS = "killer whale"

IMAGENET_INDEXES = None

SAVE_ALL = False

toolbox = None
######################## PYNEVAR territory


def teste(**kwargs):
    #print("hello!")
    # read parameters
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    objective = kwargs.get('objective')
    _resolution = kwargs.get('resolution')
    _stf = kwargs.get('stf')

    images = True
    # set objective function according to min/max
    fit = 0
    if objective == 'minimizing':
        condition = lambda: (fit < max_fit)  # minimizing
        max_fit = float('inf')
    else:
        condition = lambda: (fit > max_fit) # maximizing
        max_fit = float('-inf')

    fn = f_path + "gen" + str(generation).zfill(5)
    fitness = []
    best_ind = 0

    # scores
    for index in range(len(tensors)):
        if generation % _stf == 0:
            save_image(tensors[index], index, fn, _resolution, BGR=do_bgr) # save image
        #print(tensors[index])
        # fit = mean - std
        fit = keras_fitness([tensors[index]],256,256)[0][0]
        #print(fit)
        #print(fit[0])
        if condition():
            max_fit = fit
            best_ind = index
        fitness.append(fit)
        population[index]['fitness'] = fit

    # save best indiv
    if images:
        save_image(tensors[best_ind], best_ind, fn, _resolution, addon='_best', BGR=do_bgr)
    return population, best_ind

# if no function set is provided, the engine will use all internally available operators:
#fset = {'abs', 'add', 'and', 'cos', 'div', 'exp', 'frac', 'if', 'len', 'log',
#        'max', 'mdist', 'min', 'mod', 'mult', 'neg', 'or', 'pow', 'sign', 'sin', 'sqrt',
#        'sub', 'tan', 'warp', 'xor'}

# if no function set is provided, the engine will use all internally available operators:


def keras_fitness(individuals, img_width=256, img_height=256):
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
    # C:\Users\macha\coding\nevar-not-dead\deap-pylinhas\deap-pylinhas-engine\declarative_engine.py
    #start = time.time()
    for ind in individuals:
        #f_ = toolbox.compile(expr=ind)
        # TODO prego  - fazer um function para converter de tensor para array

        aux = convert_to_array(ind,(256,256,3))
        #print(aux.shape)
        #img_array = image_generation(ind, img_width, img_height)  # TODO: Only square images now please!
        #print(ind)
        #print(time.time() - start)

        img = Image.fromarray(aux, 'RGB')
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
        fitness_partials[k] = worthy.flatten()

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
    #merged = np.dstack(prediction_list)[0]
    return (rewards, fitness_partials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evolve to objective")
    parser.add_argument("--networks", default=NETWORKS, help="comma separated list of networks (no spaces). Default is {}.".format(NETWORKS))
    args = parser.parse_args()

    ACTIVE_MODELS = get_active_models_from_arg(NETWORKS)
    ACTIVE_MODELS_QUANTITY = len(ACTIVE_MODELS.keys())

    # NIMA likes 224 by 224 pixel images, the remaining 3 are the RBG color channels
    #resolution = [28, 28, 1]
    resolution = [256, 256, 3]

    # GP params (teste super simples)
    dev = '/gpu:0'  # device to run, write '/cpu_0' to tun on cpu
    number_generations = 500
    pop_size = 50
    tour_size = 5
    mut_prob = 0.9
    cross_prob = 0.7
    # tell the engine that the RGB does not explicitly make part of the terminal set
    edims = 2

    #fset = {'add', 'cos', 'div', 'if', 'mult', 'sign', 'sin', 'xor'}

    fset = {'abs', 'add', 'cos', 'div', 'exp', 'if',
        'max', 'mdist', 'min', 'mult', 'neg', 'sin', 'sqrt',
        'sub', 'tan', 'warp'}

    #fset = {'abs', 'add', 'cos', 'div', 'exp',
    #    'max', 'mdist', 'min', 'mult', 'neg', 'sin',
    #    'sub', 'tan'}


    # Initialize NIMA classifier
    #base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    #x = Dropout(0)(base_model.output)
    #x = Dense(10, activation='softmax')(x)
    #model = Model(base_model.input, x)
    #model.load_weights('weights/weights_mobilenet_aesthetic_0.07.hdf5')

    class_mapping = open_class_mapping()
    if TARGET_CLASS is None or TARGET_CLASS == "none":
        IMAGENET_INDEXES = None
    else:
        IMAGENET_INDEXES = get_class_index_list(class_mapping, TARGET_CLASS)

    ACTIVE_MODELS = get_active_models_from_arg(NETWORKS)
    ACTIVE_MODELS_QUANTITY = len(ACTIVE_MODELS.keys())

    for seed_i in range(0,30):
        seed = seed_i  # reproducibility
        # create engine
        engine = Engine(fitness_func=teste,
						population_size=pop_size,
						tournament_size=tour_size,
						mutation_rate=mut_prob,
						crossover_rate=cross_prob,
						target_dims=resolution,
						method='ramped half-and-half',
						objective='maximizing',
						device=dev,
						stop_criteria='generation',
						elitism=1,
						stop_value=number_generations,
						effective_dims = edims,
						seed = seed,
						debug=0,
						save_to_file=10, # save all images from each n generations
                        do_bgr=do_bgr,
						save_graphics=True,
						show_graphics=False,
						write_gen_stats=True,
						write_log = True,
						write_final_pop = True,
						read_init_pop_from_file = None,
						bloat_control = 'dynamic',
						min_tree_depth = -1,
						max_tree_depth = 12,
						max_init_depth = None,
						min_init_depth = None,
						max_subtree_dep = None,
						min_subtree_dep = None,
                        save_best = True,
						operators=fset)
                        #read_init_pop_from_file = "/home/scarlett/Documents/TensorGP/TensorGP-master/runs/run__2021_11_19__18_30_27_780__107305148598124544__images/run__2021_11_19__18_30_27_830__107305148598124544_final_pop.txt") # read predefined pop
        engine.run()


