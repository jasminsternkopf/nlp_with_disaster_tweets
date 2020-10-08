import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from plot_scores import dump_pickle, plot_test_and_train_in_axes

DATA_DIR = "data"

if __name__ == "__main__":

    dim_featurespace = list(range(1, 16))
    LSI_dimension_multiplier = 50
    parameters_SVC = {'Classifier__C': (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,
                                        10, 50, 100), 'Classifier__kernel': ('linear', 'rbf', 'poly', 'sigmoid')}

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    plot_test_and_train_in_axes(axes, 'LSI', os.path.join(DATA_DIR, 'LSI Scores'), os.path.join(DATA_DIR, 'LSI Scores Train'),
                                'LSI with SVC', 'b', dim_featurespace, os.path.join(DATA_DIR, 'LSI Params'), parameters_SVC, LSI_dimension_multiplier)
    plot_test_and_train_in_axes(axes, 'SIPLS', os.path.join(DATA_DIR, 'SIPLS Scores'), os.path.join(DATA_DIR, 'SIPLS Scores Train'),
                                'SIPLS with corresponding classification method', 'g', dim_featurespace)
    plot_test_and_train_in_axes(axes, 'LSIPLS', os.path.join(DATA_DIR, 'LSIPLS Scores'), os.path.join(DATA_DIR, 'LSIPLS Scores Train'),
                                'LSIPLS with SVC', 'r', dim_featurespace, os.path.join(DATA_DIR, 'LSIPLS Params'), parameters_SVC)

    axes[0].set_ylabel('F1 score for test data')
    axes[1].set_ylabel('F1 score for training data')
    axes[1].legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.2), fancybox=True, shadow=True, ncol=5)
    for i in range(2):
        axes[i].set_xlabel('Dimension of the projection room (*50 for LSI)', labelpad=1)
        axes[i].set_xticks(dim_featurespace)
    plt.savefig("results.pdf", bbox_inches='tight')
    dump_pickle('Di', dim_featurespace)
    plt.show()
