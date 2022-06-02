import csv
import importlib
import os
import sys

import numpy as np

from config import model_groups


def map_number(n, start1, stop1, start2, stop2):
    return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2


def create_save_folder(save_folder, sub_folder):
    found = False
    sub_backup = sub_folder
    sf = f"{save_folder}/{sub_folder}"

    i = 1
    while not found:
        if os.path.exists(sf):
            i += 1
            sub_folder = sub_backup + "_v" + str(i)
            sf = f"{save_folder}/{sub_folder}"
        else:
            found = True

    os.makedirs(sf)

    return save_folder, sub_folder


def save_gen_best(save_folder, sub_folder, experiment_name, data):
    with open(f"{save_folder}/{sub_folder}/{experiment_name}_gens_bests.txt", "a") as text_file:
        if data[0] == 0:
            print(f"gen ind fit height", file=text_file)
        genotype = '[' + ','.join(map(lambda x: str(x), np.array(data[1]))) + ']'

        print(f"{data[0]} {genotype} {data[2]} {data[3]}", file=text_file)

    with open(f"{save_folder}/{sub_folder}/{experiment_name}_gens_bests.csv", "a") as csv_file:
        csv_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if data[0] == 0:
            csv_file.writerow(['gen', 'ind', 'fit', 'height'])
        csv_file.writerow([data[0], genotype, data[2], data[3]])


def unpack_models_string(models_string):
    # a messy way to do substiution of aliases. whatever.
    cur_models_string = ""
    next_models_string = models_string
    while cur_models_string != next_models_string:
        cur_models_string = next_models_string
        if not next_models_string.endswith(","):
            next_models_string = next_models_string + ","
        for key in model_groups:
            next_models_string = next_models_string.replace(key, model_groups[key])
        # print("how about ", cur_models_string, "becoming", next_models_string)
    return cur_models_string


def unpack_requested_networks(networks):
    networks = unpack_models_string(networks)
    requested_networks = networks.split(",")
    # remove empty strings
    requested_networks = [x for x in requested_networks if x]
    # remove duplicates and sort
    requested_networks = sorted(list(dict.fromkeys(requested_networks)))
    return requested_networks


def get_model_from_name(k):
    model = load_scoring_object(k)
    return model


def get_active_models_from_arg(networks):
    requested_networks = unpack_requested_networks(networks)
    print("Requested networks: ", requested_networks)
    active_models = {}
    for k in requested_networks:
        if not k.startswith("standard"):
            print("Setting up {}".format(k))
            active_models[k] = get_model_from_name(k)
    if len(active_models) == 0:
        print("_____ WARNING: no active models ______")
    return active_models


def helpful_interface_message_exit(model_interface, e):
    print("==> Failed to load supporting class {}".format(model_interface))
    print("==> Check that package {} is installed".format(model_interface.split(".")[0]))
    print("(exception was: {})".format(e))
    sys.exit(1)


def load_scoring_object(scoring_string):
    print("Putas", scoring_string)
    scoring_parts = scoring_string.split(":")
    fullname = scoring_parts[0]
    config_suffix = ""
    if len(scoring_parts) > 1:
        config_suffix = scoring_parts[1]
    model_class_name = "Scoring"
    model_module_name = fullname
    # print("Loading {} class from {}".format(model_class_name, model_module_name))
    try:
        scoring_class = getattr(importlib.import_module(model_module_name), model_class_name)
    except ImportError:
        try:
            # fallback: try loading from "scoring" subdirectory of library path (todo: default/enforce?)
            # print("isto é meu    "+model_module_name)
            scoring_class = getattr(importlib.import_module("score." + model_module_name), model_class_name)
        except ImportError as e:
            helpful_interface_message_exit(fullname, e)
    # print("class loaded.")
    scoring_object = scoring_class(config_suffix)
    return scoring_object