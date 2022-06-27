from deap import base
from deap import cma
from deap import creator
from deap import tools
import numpy as np

from render.organic import OrganicRenderer
from config import *


def dumb_fitness(peepz):
    return [1.0]


render = OrganicRenderer()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", dumb_fitness)
# strategy = cma.Strategy(centroid=generate_individual_with_embeddings(), sigma=0.2, lambda_=args.pop_size)
strategy = cma.Strategy(centroid=np.random.normal(0.5, .5, render.genotype_size), sigma=0.5, lambda_=POP_SIZE)
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

for gen in range(5):
    COUNT_GENERATION = gen
    # Generate a new population
    population = toolbox.generate()
    # Evaluate the individuals
    COUNT_IND = 0
    COUNT_GENERATION = gen
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Update the strategy with the evaluated individuals
    toolbox.update(population)

    # Update the hall of fame and the statistics with the
    # currently evaluated population
    halloffame.update(population)
    record = stats.compile(population)
    logbook.record(evals=len(population), gen=gen, **record)

assert len(population) == POP_SIZE

print(logbook)
