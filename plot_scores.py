import pickle
from os import path

import numpy as np

from get_scores import get_score_vector


def open_pickle(file_name: str):
    if path.exists(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    else:
        return None


def dump_pickle(file_name: str, data):
    with open(file_name, "wb") as file:
        pickle.dump(data, file)


def plot_test_and_train_in_axes(axes, transformer, test_score_file, train_score_file, label, color, dims, params_file=None, parameters_SVC=None, dim_multiplier=1):
    """
    Loads the saved test and training scores, if they exist, and the stored dimensions of the featurespace
    For LSI, it is intended to pass dim_multiplier - the number with which the entries in dims should be multiplied
    If the files are empty or, in case of dimension reduction, the lengths of the stored score vectors do not fit the lenght of the vector containing the desired dimensions, the scores will be freshly computed, stored and then plotted, otherwise the stored scores will be plotted
    """
    test_score = open_pickle(test_score_file)
    train_score = open_pickle(train_score_file)
    if (transformer is not None and (test_score is None or train_score is None or len(dims) != len(test_score) or len(dims) != len(train_score))) or (transformer is None and (test_score is None or train_score is None)):
        test_score, train_score, best_params = get_score_vector(
            transformer, dim_multiplier * np.array(dims), parameters_SVC)
        dump_pickle(test_score_file, test_score)
        dump_pickle(train_score_file, train_score)
        if params_file != None:
            dump_pickle(params_file, best_params)
    if transformer is None:  # in the baseline scenario are no projection rooms
        test_score = test_score * np.ones(len(dims))
        train_score = train_score * np.ones(len(dims))
    axes[0].plot(dims, test_score, color=color)
    axes[1].plot(dims, train_score, label=label, color=color)
