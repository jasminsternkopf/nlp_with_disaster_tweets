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


def plot_test_and_train_in_axes(axes, transformer: str, test_score_file, train_score_file, label, color, dims=None, params_file=None, parameters_SVC=None, dim_multiplier=1):
    """
    Loads the saved test and training scores, if they exist, and the stored dimensions of the featurespace
    For LSI, it is neccessary to pass the desired dimensions of the featurespace, otherwise they will be 1 to the length of the saved test score vector
    If you want to compute the scores for different dimensions than 1 to the length of the saved test score vector, it is neccessary to pass dims
    If the files are empty or the lengths of the stored score vectors do not fit the lenght of the vector containing the desired dimensions, the scores will be freshly computed, stored and then plotted, otherwise the stored scores will be plotted
    """
    test_score = open_pickle(test_score_file)
    train_score = open_pickle(train_score_file)
    # stored_dim_featurespace = open_pickle('Dimension Featurespace')
    # if dims == None:  # if test_score_file does not exist, it is neccessary to übergeben dims
    #    dims = range(1, len(test_score) + 1)
    # if test_score == None or train_score == None or len(stored_dim_featurespace) != len(dims) or len(dims) != len(test_score) or len(dims) != len(train_score):
    # Zwischenlösung, .any funktioniert nicht für None
    if test_score.any == None or train_score.any == None or len(dims) != len(test_score) or len(dims) != len(train_score):
        test_score, train_score, best_params = get_score_vector(
            transformer, dim_multiplier * np.array(dims), parameters_SVC)
        dump_pickle(test_score_file, test_score)
        dump_pickle(train_score_file, train_score)
        if params_file != None:
            dump_pickle(params_file, best_params)
    axes[0].plot(dims, test_score, color=color)
    axes[1].plot(dims, train_score, label=label, color=color)
