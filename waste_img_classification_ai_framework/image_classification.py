from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import zipfile as zf
import shutil
import re
import seaborn as sns
import random
from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from .live_img_capture import *


import logging
logging.basicConfig(filename='waste_image_processing.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)



def main(path):
    #capture frames from camara
    image_frame_list=capture_live_img()
    write_img(image_frame_list, path)
    os.chdir(path)
    files = zf.ZipFile('waste-imge-dataset.zip', 'r')
    files.extractall()
    files.close()

    img=os.listdir(os.path.join(os.getcwd(), "waste-imge-dataset"))
    image_subsets = ['traing', 'validation']
    waste_types = ['glass', 'metal', 'paper', 'plastic', 'trash','cardboard']

    ## create destination folders for data subset and waste type
    for image_subset in image_subsets:
        for waste_type in waste_types:
            img_directory = os.path.join('data', image_subset, waste_type)
            if not os.path.exists(img_directory):
                os.makedirs(img_directory)

    if not os.path.exists(os.path.join('data', 'test_data')):
        os.makedirs(os.path.join('data', 'test_data'))

    #Spliting the data set for train, validate & test
    for waste_type in waste_types:
        source_folder = os.path.join('waste-imge-dataset', waste_type)
        train_ind, valid_ind, test_ind = split_indices(source_folder, 1, 1)

        train_names = get_names(waste_type, train_ind)
        train_source_files = [os.path.join(source_folder, name) for name in train_names]
        train_dest = "data/traing_data/" + waste_type
        move_files(train_source_files, train_dest)

        ## move source files to valid
        valid_names = get_names(waste_type, valid_ind)
        valid_source_files = [os.path.join(source_folder, name) for name in valid_names]
        valid_dest = "data/validation_data/" + waste_type
        move_files(valid_source_files, valid_dest)

        ## move source files to test
        test_names = get_names(waste_type, test_ind)
        test_source_files = [os.path.join(source_folder, name) for name in test_names]
        ## I use data/test here because the images can be mixed up
        move_files(test_source_files, "data/test_data")

        ## get a path to the folder with images
        path = Path(os.getcwd()) / "data"

        tfms = get_transforms(do_flip=True, flip_vert=True)
        data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=16)

        data.show_batch(rows=4, figsize=(10, 8))

        learn = create_cnn(data, models.resnet34, metrics=error_rate)

        learn.lr_find(start_lr=1e-6, end_lr=1e1)
        learn.recorder.plot()

        learn.fit_one_cycle(20, max_lr=5.13e-03)

        interp = ClassificationInterpretation.from_learner(learn)
        losses, idxs = interp.top_losses()

        interp.plot_top_losses(9, figsize=(15, 11))

        doc(interp.plot_top_losses)
        interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

        preds = learn.get_preds(ds_type=DatasetType.Test)

def split_indices(directory, seed1, seed2):
    n = len(os.listdir(directory))
    full_set = list(range(1, n + 1))
    random.seed(seed1)
    train = random.sample(list(range(1, n + 1)), int(.5 * n))
    remain = list(set(full_set) - set(train))
    random.seed(seed2)
    valid = random.sample(remain, int(.5 * len(remain)))
    test = list(set(remain) - set(valid))

    return (train, valid, test)


def get_names(waste_type, indices):
    file_names = [waste_type + str(i) + ".jpg" for i in indices]
    return (file_names)


def move_files(source_files, destination_folder):
    for file in source_files:
        shutil.move(file, destination_folder)


if __name__ == '__main__':
    #Data location path
    filepath=os.path.dirname(os.path.realpath(__file__))
    filepath=os.path.join(filepath,'data_set')
    main(filepath)

