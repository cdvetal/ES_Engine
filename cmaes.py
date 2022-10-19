import pickle
import random

import numpy as np
import torch
from deap import base
from deap import cma
from deap import creator
from deap import tools
from torch import optim
from torchvision.utils import save_image

from fitnesses import calculate_fitness
from utils import save_gen_best

cur_iteration = 0


def evaluate(args, individual):
    renderer = args.renderer

    optimizers = renderer.to_adam(individual)

    img = renderer.render()
    fitness = calculate_fitness(args.fitnesses, img)

    for gen in range(args.adam_steps):
        for optimizer in optimizers:
            optimizer.zero_grad()

        (-fitness).backward()

        for optimizer in optimizers:
            optimizer.step()

        img = renderer.render()
        fitness = calculate_fitness(args.fitnesses, img)

        if args.renderer_type == "vdiff" and gen >= 1:
            lr = renderer.sample_state[6][gen] / renderer.sample_state[5][gen]
            individual = renderer.makenoise(gen, individual)
            individual.requires_grad_()
            individual = [individual]
            optimizer = optim.Adam(individual, lr=min(lr * 0.001, 0.01))

        if torch.min(img) < 0.0:
            img = (img + 1) / 2

        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{cur_iteration}_{gen}.png")

    print(fitness)

    if args.lamarck:
        individual[:] = renderer.get_individual()

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return [fitness]


def main_cma_es(args):
    global cur_iteration

    renderer = args.renderer

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, args)
    strategy = cma.Strategy(centroid=renderer.generate_individual(), sigma=args.sigma, lambda_=args.pop_size)
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
                _ = renderer.to_adam(ind, gradients=False)
                img = renderer.render()
                # img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_{index}.png")
                if torch.min(img) < 0.0:
                    img = (img + 1) / 2
                save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_{index}.png")

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
            print("Best individual:", halloffame[0].fitness.values)
            _ = renderer.to_adam(halloffame[0], gradients=False)
            img = renderer.render()
            if torch.min(img) < 0.0:
                img = (img + 1) / 2
            save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_best.png")

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
