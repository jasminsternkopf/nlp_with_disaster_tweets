import pickle
from os import path

import numpy as np
from matplotlib import pyplot as plt

from get_scores import get_score_vector

if __name__ == "__main__":

    dim_featurespace = list(range(1, 16))
    fig, ax = plt.subplots()
    LSI_dimension_multiplier = 50

    def open_pickle(file_name: str):
        if path.exists(file_name):
            with open(file_name, 'rb') as file:
                return pickle.load(file)
        else:
            return None

    def dump_pickle(file_name: str, data):
        with open(file_name, "wb") as file:
            pickle.dump(data, file)

    def plot_test_and_train_in_axes(axes, transformer: str, test_score_file, train_score_file, label, color, dims=None, params_file=None, parameters_SVC=None):
        """
        Loads the saved test and training scores, if they exist, and the stored dimensions of the featurespace
        For LSI, it is neccessary to pass the desired dimensions of the featurespace, otherwise they will be 1 to the length of the saved test score vector
        If you want to compute the scores for different dimensions than 1 to the length of the saved test score vector, it is neccessary to pass dims
        If the files are empty or the lengths of the stored score vectors do not fit the lenght of the vector containing the desired dimensions, the scores will be freshly computed, stored and then plotted, otherwise the stored scores will be plotted
        """
        test_score = open_pickle(test_score_file)
        train_score = open_pickle(train_score_file)
        #stored_dim_featurespace = open_pickle('Dimension Featurespace')
        if dims == None:  # if test_score_file does not exist, it is neccessary to übergeben dims
            dims = range(1, len(test_score) + 1)
        # if test_score == None or train_score == None or len(stored_dim_featurespace) != len(dims) or len(dims) != len(test_score) or len(dims) != len(train_score):
        if test_score == None or train_score == None or len(dims) != len(test_score) or len(dims) != len(train_score):
            test_score, train_score, best_params = get_score_vector(
                transformer, dims, parameters_SVC)
            dump_pickle(test_score_file, test_score)
            dump_pickle(train_score_file, train_score)
            if params_file != None:
                dump_pickle(params_file, best_params)
        axes[0].plot(dims, test_score, color=color)
        axes[1].plot(dims, train_score, label=label, color=color)

    with open("Dimension Featurespace", "rb") as dim_featurespace_file:
        stored_dim_featurespace = pickle.load(dim_featurespace_file)

    with open("Dimension Featurespace", "wb") as dim_featurespace_file_new:
        pickle.dump(dim_featurespace, dim_featurespace_file_new)

    parameters_SVC = {'Classifier__C': (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,
                                        10, 50, 100), 'Classifier__kernel': ('linear', 'rbf', 'poly', 'sigmoid')}

    ax.set_xlabel('Dimension of the projection room (*50 for LSI)')
    ax.set_ylabel('F1 score')
    ax.set_xticks(dim_featurespace)
    ax.legend()
    plt.savefig("results.pdf", bbox_inches='tight')
    dump_pickle('Di', dim_featurespace)
    plt.show()
