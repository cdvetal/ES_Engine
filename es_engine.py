import os

from utils import save_gen_best, create_save_folder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from deap import base
from deap import cma
from deap import creator
from deap import tools
import torch
import argparse


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


def generate_individual_with_embeddings():
    latent = torch.nn.Parameter(torch.zeros(BATCH_SIZE, 128).normal_(std=1).float().cuda())
    params_other = torch.zeros(BATCH_SIZE, 1000).normal_(-3.9, .3).cuda()
    classes = torch.sigmoid(torch.nn.Parameter(params_other))
    embed = big_sleep_cma_es.model.embeddings(classes)
    cond_vector = torch.cat((latent, embed), dim=1)
    ind = cond_vector.cpu().detach().numpy().flatten()
    # cond_vector = big_sleep_cma_es.CondVectorParameters(ind, batch_size=BATCH_SIZE)
    # big_sleep_cma_es.save_individual_cond_vector(cond_vector, f"PONTO_INICIAL.png")
    return ind


def main(args):
    print(args)
    "
    # The cma module uses the np random number generator
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    # strategy = cma.Strategy(centroid=np.random.normal(0.5, .5, GENOTYPE_SIZE), sigma=0.5, lambda_=POP_SIZE)
    strategy = cma.Strategy(centroid=generate_individual_with_embeddings(), sigma=0.2, lambda_=POP_SIZE)
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
    parser = argparse.ArgumentParser(description="evolve to objective")

    parser.add_argument('--random-seed', default=0, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(0))
    parser.add_argument('--save-folder', default="experiments", help="Directory to experiment outputs. Default is 'experiments'.")
    parser.add_argument('--n-gens', default=125, type=int, help='Maximum generations. Default is 125.')
    parser.add_argument('--pop-size', default=10, type=int, help='Population size. Default is 10.')
    parser.add_argument('--save-all', default=False, action='store_true', help='Save all Individual images. Default is False.')
    parser.add_argument('--lamarck', default=False, action='store_true', help='Lamarckian evolution')
    parser.add_argument('--verbose', default=False, action='store_true', help='Lamarckian evolution')
    args = parser.parse_args()

    args.sub_folder = f"{args.n_gens}_{args.pop_size}"

    return args


if __name__ == "__main__":
    args = setup_args()
    main(args)
