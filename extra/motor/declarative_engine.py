from datetime import datetime
import keyboard
import random
import time
import warnings
import numpy as np
import os
import pickle
import big_sleep
from extra_tools import save_gen_best, create_save_folder, save_experimental_setup, save_logbook_as_csv
from PIL import Image
from deap import gp, creator, base, tools, algorithms
import argparse
# from profiler import profile
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from itertools import repeat


IMG_SIZE = IMG_WIDTH, IMG_HEIGHT = (512, 512)  # ATTENTION!!!! Only square images now please.


# ---------------------------
CROSSOVER_PROBABILITY1 = 0.6
CROSSOVER_PROBABILITY2 = 0.6

# ---------------------------



#configuration of the latent space
LATENT_SPACE_SIZE = 128
IMAGE_NET_CLASSES = 1000
BATCH_SIZE = 1
ALL_DIFFERENT = True

POP_SIZE = 10
N_GENS = 2
ELITISM = 1
CXPB = 1
MUTPB = 1
# Parameters for Gaussian Mutation
MUT_MU1 = 0
MUT_SIGMA1 = 1
MUT_MU2 = 0
MUT_SIGMA2 = 1

MUT_INDPB = 0.2

TOURNAMENT_SIZE = 5

TARGET_FITNESS = 100000,
RANDOM_SEED = None

CHECKPOINT_FREQ = 10
CHECKPOINT = None  # None or "Experiment_name.pkl""

COUNTER = 0
SAVE_ALL = False




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


#Penousal: verificar
def myswap(lower, upper, ind1,ind2):
    cxpoint1 = random.randint(lower, upper)
    cxpoint2 = random.randint(lower, upper - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2


#Penousal: verificar
def cxTwoPointCopyNew(ind1, ind2):  # funcao de crossover
    # vou tratar como se fossem dois cromossomas distintos
    # a percentagem de crossover vai passar a 1 (entra sempre aqui)
    # aqui tenho a probabilidade para o primeiro cromossoma (lats) e para o segundo (classes)
    # mais tarde estes parametros devem vir da linha de comando
    #crossover nas lats com probabilidade de 0.6
    global LATENT_SPACE_SIZE, CROSSOVER_PROBABILITY1,CROSSOVER_PROBABILITY2
    if (random.random()<CROSSOVER_PROBABILITY1):
        ind1,ind2 = myswap(1,LATENT_SPACE_SIZE, ind1,ind2)

    #crossover nas calsses com probabilidade de 0.6
    if (random.random()<CROSSOVER_PROBABILITY2):
        size = len(ind1)
        ind1,ind2 = myswap(LATENT_SPACE_SIZE,size,ind1,ind2)
    return ind1, ind2



#Penousal: verificar
def mutGaussianLimited(individual, mu1, mu2, sigma1,sigma2, indpb):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    if not isinstance(mu1, Sequence):  #se bem percebo posso ter uma probabilidade independente para cada gene, está giro está
        mu1 = repeat(mu1, size)
    elif len(mu1) < size:
        raise IndexError("mu1 must be at least the size of individual: %d < %d" % (len(mu1), size))
    if not isinstance(sigma1, Sequence):
        sigma1 = repeat(sigma1, size)
    elif len(sigma1) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma1), size))
    if not isinstance(mu2, Sequence):
        mu2 = repeat(mu2, size)
    elif len(mu2) < size:
        raise IndexError("mu2 must be at least the size of individual: %d < %d" % (len(mu2), size))
    if not isinstance(sigma2, Sequence):
        sigma2 = repeat(sigma2, size)
    elif len(sigma2) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma2), size))
        
#acho que não estamos a preservar as distribuições...
    for i, m1, m2, s1, s2 in zip(range(size), mu1, mu2, sigma1, sigma2):  #isto parece profundamente estupido, mas ok
        if random.random() < indpb:
            if (i<LATENT_SPACE_SIZE):
                individual[i] += random.gauss(m1, s1)
                if (individual[i] <-3): ##mais de 3 vezes o desvio padrão PREGO 
                    individual[i] =-3  #devia ir buscar o std a outro sitio qq mas acho que isso nem definido está, note-se que é o STD do initIndividual e não da mutação
                if (individual[i] > 3):
                    individual[i] = 3
            else:
                individual[i] += random.gauss(m2, s2)
                if (individual[i] < -6.5 ):   ##mais de 3 vezes o desvio padrão PREGO #devia ir buscar o std a outro sitio qq, 
                    individual[i] = -6.5
                if (individual[i] > 1.5):
                    individual[i] = 1.5
    return individual,



#fitness para floats
def clip_fitness(individual, img_width, img_height):
    import big_sleep
    global COUNTER
    ind_array = np.array(individual)
    latent_space = big_sleep.Pars(ind_array,all_different=ALL_DIFFERENT, batch_size=BATCH_SIZE) #
    result = big_sleep.evaluate(latent_space)


        #if all_different:
        #    latent_space_numpy = ind_numpy[ : 128 * batch_size].reshape(16,128)
        #    imagenet_classes_numpy =  ind_numpy[128 * batch_size : ].reshape(16,1000) 
        #    self.normu = torch.nn.Parameter(torch.tensor(latent_space_numpy).float().cuda())
        #    self.cls = torch.nn.Parameter(torch.tensor(imagenet_classes_numpy).float().cuda())
        #  else:
        #    latent_space_numpy = ind_numpy[: 128]
        #    imagenet_classes_numpy =  ind_numpy[128 : ]
        #    self.normu = torch.nn.Parameter(torch.tensor(latent_space_numpy).repeat(32,1).float().cuda())
        #    self.cls = torch.nn.Parameter(torch.tensor(imagenet_classes_numpy).repeat(32,1).float().cuda())


    # print(float(result[0].float().cpu()) * -1)
    # print(float(result[1].float().cpu()) * -1)
    # print(float(result[2].float().cpu()) * -1)
    # input()
    # print("demonio")
    # print(float(result[2].float().cpu()))
    # big_sleep.checkin(result, latent_space)
    if (COUNTER%320 == 0):
        print("Lat loss", float(result[0].float().cpu()) * -1)
        print("Class loss", float(result[1].float().cpu()) * -1)
        print("Similarity", float(result[2].float().cpu()) * -1)
    COUNTER +=1
    big_sleep.inc()
    #return float(result[2].float().cpu()) * -1,
    return (float(result[0].float().cpu()) * -1) *0 + (float(result[1].float().cpu()) * -1) *0 + (float(result[2].float().cpu()) * -1)*1,



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
            ind.fitness.values = eval_result
        
        if SAVE_ALL:
            for index, ind in enumerate(population):
                latent_space = big_sleep.Pars(np.array(ind),all_different=ALL_DIFFERENT, batch_size=BATCH_SIZE)
                big_sleep.save_individual(latent_space, f"{save_folder}/{sub_folder}/{experiment_name}_0_{index}.png")
        
        if halloffame is not None:
            halloffame.update(population)
            save_gen_best(save_folder, sub_folder, experiment_name, [0, halloffame[0], halloffame[0].fitness.values, "_"])
            latent_space = big_sleep.Pars(np.array(halloffame[0]),all_different=ALL_DIFFERENT, batch_size=BATCH_SIZE)
            big_sleep.save_individual(latent_space, f"{save_folder}/{sub_folder}/{experiment_name}_0_best.png")

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
        #best = max(population, key=attrgetter("fitness"))
        gen += 1
        # Select the next generation individuals
        selected = toolbox.select(population, POP_SIZE)
        offspring = [toolbox.clone(ind) for ind in selected]

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        offspring[:] = tools.selBest(population, ELITISM) + offspring[ELITISM:] ## penousal: adiciono a elite aos offspring 

        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        invalid_ind = [ind for ind in offspring if True] #penousal: vai tudo ser reavaliado (sim eu sei que não preciso disto assim)

        eval_results = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, eval_result in zip(invalid_ind, eval_results):
            ind.fitness.values = eval_result

        if SAVE_ALL:
            for index, ind in enumerate(offspring):
                latent_space = big_sleep.Pars(np.array(ind),all_different=ALL_DIFFERENT, batch_size=BATCH_SIZE)
                big_sleep.save_individual(latent_space, f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_{index}.png")
        if keyboard.is_pressed('s'):  ##penousal v~e aqui nuno
            print('Saving all')
            for index, ind in enumerate(offspring):
                latent_space = big_sleep.Pars(np.array(ind),all_different=ALL_DIFFERENT, batch_size=BATCH_SIZE)
                big_sleep.save_individual(latent_space, f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_{index}.png")

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            save_gen_best(save_folder, sub_folder, experiment_name, [gen, halloffame[0], halloffame[0].fitness.values, "_"])
            latent_space = big_sleep.Pars(np.array(halloffame[0]),all_different=ALL_DIFFERENT, batch_size=BATCH_SIZE)
            big_sleep.save_individual(latent_space, f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_best.png")

        # Replace the current population by the offspring
        #population[:] = tools.selBest(population, ELITISM) + offspring[ELITISM:]

        population[:] = offspring[:]



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


def initIndividual(pcls):
    global IMAGE_NET_CLASSES, LATENT_SPACE_SIZE
    latent = np.random.normal(0,1, LATENT_SPACE_SIZE * BATCH_SIZE )
    classes =  np.random.normal(-3.9, .3, IMAGE_NET_CLASSES * BATCH_SIZE)
    ind = pcls(np.concatenate((latent, classes), axis = None))
    return ind

# @profile
def main():
    global CROSSOVER_PROBABILITY1, CROSSOVER_PROBABILITY2, POP_SIZE, IMG_SIZE, IMG_WIDTH, IMG_HEIGHT, N_GENS, RANDOM_SEED, CHECKPOINT, CHECKPOINT_FREQ, SAVE_ALL, TOURNAMENT_SIZE, CXPB, MUTPB, MUT_MU1, MUT_SIGMA1, MUT_INDPB, MUT_MU2, MUT_SIGMA2

    parser = argparse.ArgumentParser(description="evolve to objective")
    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is 'experiments'.")
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--max-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))
    parser.add_argument('--img-size', default=IMG_SIZE[0], type=int, help='Image dimensions during testing. Default is {}.'.format(IMG_SIZE[0]))
    parser.add_argument('--checkpoint-freq', default=CHECKPOINT_FREQ, help='Checkpoint backup frequency. Default is every {} generations.'.format(CHECKPOINT_FREQ))
    parser.add_argument('--from-checkpoint', default=CHECKPOINT, help='Checkpoint file from which you want to continue evolving. Default is {}.'.format(CHECKPOINT))
    parser.add_argument('--save-all', default=SAVE_ALL, action='store_true', help='Save all Individual images. Default is {}.'.format(SAVE_ALL))
    parser.add_argument('--tournament-size', default=TOURNAMENT_SIZE, type=int, help='Tournament size for selection operator. Default is {}.'.format(TOURNAMENT_SIZE))
    parser.add_argument('--cxpb1', default=CROSSOVER_PROBABILITY1, type=float, help='Crossover probability. Default is {}.'.format(CROSSOVER_PROBABILITY1))
    parser.add_argument('--cxpb2', default=CROSSOVER_PROBABILITY2, type=float, help='Crossover probability. Default is {}.'.format(CROSSOVER_PROBABILITY2))
    parser.add_argument('--mutpb', default=MUTPB, type=float, help='Mutation probability. Default is {}.'.format(MUTPB))
    parser.add_argument('--mut-mu1', default=MUT_MU1, type=float, help='Mean or python:sequence of means for the gaussian addition mutation. Default is {}.'.format(MUT_MU1))
    parser.add_argument('--mut-sigma1', default=MUT_SIGMA1, type=float, help='Standard deviation or python:sequence of standard deviations for the gaussian addition mutation. Default is {}.'.format(MUT_SIGMA1))
    parser.add_argument('--mut-mu2', default=MUT_MU2, type=float, help='Mean or python:sequence of means for the gaussian addition mutation. Default is {}.'.format(MUT_MU2))
    parser.add_argument('--mut-sigma2', default=MUT_SIGMA2, type=float, help='Standard deviation or python:sequence of standard deviations for the gaussian addition mutation. Default is {}.'.format(MUT_SIGMA2))
    parser.add_argument('--mut-indpb', default=MUT_INDPB, type=float, help='Independent probability for each attribute to be mutated. Default is {}.'.format(MUT_INDPB))

    args = parser.parse_args()

    save_folder = args.save_folder
    POP_SIZE = int(args.pop_size)
    N_GENS = int(args.max_gens)
    RANDOM_SEED = args.random_seed
    CHECKPOINT = args.from_checkpoint
    CHECKPOINT_FREQ = int(args.checkpoint_freq)
    SAVE_ALL = args.save_all
    TOURNAMENT_SIZE = args.tournament_size
    CROSSOVER_PROBABILITY1 = args.cxpb1
    CROSSOVER_PROBABILITY2 = args.cxpb2
    MUTPB = args.mutpb
    # Parameters for Gaussian Mutation
    MUT_MU1 = args.mut_mu1
    MUT_SIGMA1 = args.mut_sigma1
    MUT_MU2 = args.mut_mu2
    MUT_SIGMA2 = args.mut_sigma2
    MUT_INDPB = args.mut_indpb



    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, fitness_partials=dict)

    toolbox = base.Toolbox()
    # toolbox.register("attr_latent", np.random.normal, 0, 1)
    # toolbox.register("attr_imagenet", np.random.normal, -3.9, .3)
    toolbox.register("individual", initIndividual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxTwoPointCopyNew)
    toolbox.register("mutate", mutGaussianLimited, mu1=MUT_MU1, mu2=MUT_MU1, sigma1=MUT_SIGMA1, sigma2=MUT_SIGMA2, indpb=MUT_INDPB)

    # toolbox.decorate("mate", checkBounds(0.02, 0.98)) <-- pode ser mais elegante
    # toolbox.decorate("mutate", checkBounds(0.02, 0.98))

    toolbox.register("clip", np.clip, a_min=0.02, a_max=0.98)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("evaluate", clip_fitness, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    if CHECKPOINT:
        experiment_name = CHECKPOINT.replace("_checkpoint.pkl", "")
        # save_folder = f"experiments/{experiment_name}"
        # CHECKPOINT = f"{save_folder}/{CHECKPOINT}"
        # save_folder = "{}/{}".format(save_folder, experiment_name)
        sub_folder = "from_checkpoint"
        save_folder, sub_folder = create_save_folder(save_folder, sub_folder)
        CHECKPOINT = "{}/{}".format(save_folder, CHECKPOINT)
    else:
        experiment_name = f"{big_sleep.frase}_clip_L{LATENT_SPACE_SIZE}_{RANDOM_SEED if RANDOM_SEED else datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        sub_folder = f"{experiment_name}_{N_GENS}_{POP_SIZE}"
        save_folder, sub_folder = create_save_folder(save_folder, sub_folder)


    if args.random_seed:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        # TODO: not do this or maybe there is a tf2 way?

    pop = toolbox.population(n=POP_SIZE)
    hof = hof = tools.HallOfFame(1, similar=np.array_equal)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    # stats_mobilenet = tools.Statistics(lambda ind: ind.fitness_partials["mobilenetv2"])
    # stats_mobilenetv2 = tools.Statistics(lambda ind: ind.fitness_partials["mobilenetv2"])
    # mstats = tools.MultiStatistics(fitness=stats_fit, mobilenet=stats_mobilenet, mobilenetv2=stats_mobilenetv2)

    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    

    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    save_experimental_setup(save_folder, sub_folder, experiment_name, args)

    pop, logbook, hof = eaSimple(pop, toolbox, CXPB, MUTPB, N_GENS, mstats, halloffame=hof, verbose=True, experiment_name=experiment_name, save_folder=save_folder, sub_folder=sub_folder, checkpoint=CHECKPOINT)  # , checkpoint=".pkl"


if __name__ == "__main__":
    main()


