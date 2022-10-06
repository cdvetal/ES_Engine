import os
import pickle
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL.Image import Resampling
from deap import base
from deap import cma
from deap import creator
from deap import tools
from torch import optim
from torchvision.utils import save_image

from utils import save_gen_best

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
    # Calculate clip similarity
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_t = trans(img).unsqueeze(0).to(args.device)
    image_features = args.clip.encode_image(img_t)
    image_clip_loss = torch.cosine_similarity(args.image_features, image_features, dim=1)
    image_clip_loss *= 100

    return image_clip_loss


def fitness_clip_prompts(args, img):
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


def calculate_fitness(args, img):
    losses = []

    if args.clip_influence < 1.0 and len(args.active_models) > 0:
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

    return final_loss


def evaluate(args, individual):
    # TODO - Renderers - line_sketch
    # TODO - Test Adam optimize all losses and combinations

    renderer = args.renderer

    ind = renderer.to_adam(individual)

    optimizer = optim.Adam(ind, lr=args.lr)

    img = renderer.render(ind)
    final_loss = calculate_fitness(args, img)

    for gen in range(args.adam_steps):
        optimizer.zero_grad()
        (-final_loss).backward()
        optimizer.step()

        img = renderer.render(ind)
        final_loss = calculate_fitness(args, img)

        if args.renderer_type == "vdiff" and gen >= 1:
            lr = renderer.sample_state[6][gen] / renderer.sample_state[5][gen]
            individual = renderer.makenoise(gen, individual)
            individual.requires_grad_()
            individual = [individual]
            optimizer = optim.Adam(individual, lr=min(lr * 0.001, 0.01))

        if torch.min(img) < 0.0:
            print("Needs reverse normalization")
            img = (img + 1) / 2

        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{cur_iteration}_{gen}.png")

    print(final_loss)

    if args.lamarck:
        individual[:] = renderer.get_individual(ind)

    # print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))
    # return [(rewards[0],), fitness_partials]
    return [final_loss]


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
                img_array = renderer.chunks(ind)
                img = renderer.render(img_array)
                # img.save(f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{gen}_{index}.png")
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
            ind = renderer.to_adam(halloffame[0])
            img = renderer.render(ind)
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
