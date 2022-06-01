import csv
import os
import numpy as np
from PIL import Image
import pandas as pd
from functools import reduce
from operator import add, itemgetter

IMG_SIZE = IMG_WIDTH, IMG_HEIGHT = (256, 256)  # ATTENTION!!!! Only square images now please.


r = np.transpose(np.indices((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.int), (2, 1, 3, 0))
r = r[:, :, :, 0:2]
pointN = r.reshape(IMG_SIZE[0] * IMG_SIZE[1], 3, 2)
x = pointN[:, :, 0]
y = pointN[:, :, 1]
x = (x / IMG_SIZE[0]) * 2 - 1
y = (y / IMG_SIZE[1]) * 2 - 1


def save_img(save_folder, sub_folder, experiment_name, toolbox, gen, ind, img_width, img_height, ind_number):
    f_ = toolbox.compile(expr=ind)
    img_array = image_generation(f_, img_width, img_height)
    #print("Image genration", time.time() - start)
    # input()
    #start = time.time()
    img = Image.fromarray(img_array, 'RGB')
    if ind_number == "best":
        img.save(f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_best.png")
    else:
        img.save(f"{save_folder}/{sub_folder}/{experiment_name}_{gen}_{ind_number}.png")

    #print(time.time() - start)



def image_generation(f_, img_width, img_height):
    #print("largura"  + img_width)
    #print("altura" = img_height)
    r = f_(x, y)
    #print(r)
    r = np.nan_to_num(r, nan=1, posinf=10000, neginf=-100000)
    r[np.where(r < -1)] = -1
    r[np.where(r >= 1)] = 1 #r[np.where(r >= 1)]/(r[np.where(r >= 1)]+1)
    
    #r[np.where(r < 0)] = r[np.where(r < -0)]/(r[np.where(r < -0)]*(-1)+1)
    #r[np.where(r >= 0)] = r[np.where(r >= 0)]/(r[np.where(r >= 0)]+1)
    

    r = np.reshape(r, (IMG_SIZE[0], IMG_SIZE[1], 3))

    ##print(r.min())

    r = ((r+1)*255/2).astype(np.uint8) 
    
    # return numpy.ndarray.tolist(r)
    return r


def create_save_folder(save_folder, sub_folder):
    found = False
    sub_backup = sub_folder
    sf = f"{save_folder}/{sub_folder}"

    i = 1
    while not found:
        if os.path.exists(sf):
            i += 1
            sub_folder = sub_backup+"_v"+str(i)
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


def save_experimental_setup(save_folder, sub_folder, experiment_name, data):
    with open(f"{save_folder}/{sub_folder}/{experiment_name}_experimental_setup.txt", "a") as text_file:
        print(data, file=text_file)


def save_logbook_as_csv(save_folder, sub_folder, experiment_name, logbook):

    chapter_keys = logbook.chapters.keys()
    sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]

    data = [list(map(itemgetter(*skey), chapter)) for skey, chapter in zip(sub_chaper_keys, logbook.chapters.values())]
    data = np.array([[*a, *b] for a, b in zip(*data)])

    #data = np.array(*data)

    columns = reduce(add, [["_".join([x, y]) for y in s] for x, s in zip(chapter_keys, sub_chaper_keys)])
    # print(columns)
    # print(data)
    df = pd.DataFrame(data, columns=columns)

    keys = logbook[0].keys()
    data = [[d[k] for d in logbook] for k in keys]
    for d, k in zip(data, keys):
        df[k] = d

    df.to_csv(f"{save_folder}/{sub_folder}/{experiment_name}_logbook.csv", index=False)
