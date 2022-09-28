import os
import random
from datetime import datetime
from time import time

import clip
from PIL import Image

from adam import main_adam
from cmaes import main_cma_es
from utils import create_save_folder, get_active_models_from_arg, open_class_mapping, \
    get_class_index_list

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import torch
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
    # "vdiff": VDiffRenderer,
    "biggan": BigGANRenderer,
}


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
    parser.add_argument('--adam-steps', default=ADAM_STEPS, help='Number of steps from Adam. Default is {}.'.format(ADAM_STEPS))
    parser.add_argument('--lr', default=LR, help='Learning rate for the Adam optimizer. Default is {}.'.format(LR))
    parser.add_argument('--lamarck', default=LAMARCK, help='Lamarck. Default is {}.'.format(LAMARCK))

    args = parser.parse_args()

    args.clip_prompts = "Darth Vader"

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
    main_adam(args)
    # main_cma_es(args)
    # Get end time
    end_time = time()

    total_time = (end_time - start_time_total)
    evo_time = (end_time - start_time_evo)

    print("-" * 20)
    print("Evolution elapsed time: {:.3f}".format(evo_time))
    print("Total elapsed time: {:.3f}".format(total_time))
    print("-" * 20)
