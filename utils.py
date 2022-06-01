import csv
import os

import numpy as np


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
