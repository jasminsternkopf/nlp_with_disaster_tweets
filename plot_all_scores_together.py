import os

from matplotlib import pyplot as plt

from global_parameters import (DATA_DIR, LSI_DIMENSION_MULTIPLIER, MAX_DIM,
                               PARAMETERS_SVC)
from plot_scores import plot_test_and_train_in_axes


def plot_all_model_scores_together():
    dim_featurespace = list(range(1, MAX_DIM))
    _, axes = plt.subplots(2, 1, figsize=(8, 6))

    plot_test_and_train_in_axes(axes, None, os.path.join(DATA_DIR, 'SVC Score'), os.path.join(DATA_DIR, 'SVC Score Train'),
                                'SVC', 'y', dim_featurespace, os.path.join(DATA_DIR, 'SVC Params'), PARAMETERS_SVC)
    plot_test_and_train_in_axes(axes, 'LSI', os.path.join(DATA_DIR, 'LSI Scores'), os.path.join(DATA_DIR, 'LSI Scores Train'),
                                'LSI with SVC', 'b', dim_featurespace, os.path.join(DATA_DIR, 'LSI Params'), PARAMETERS_SVC, LSI_DIMENSION_MULTIPLIER)
    plot_test_and_train_in_axes(axes, 'SIPLS', os.path.join(DATA_DIR, 'SIPLS Scores'), os.path.join(DATA_DIR, 'SIPLS Scores Train'),
                                'SIPLS with corresponding classification method', 'g', dim_featurespace)
    plot_test_and_train_in_axes(axes, 'LSIPLS', os.path.join(DATA_DIR, 'LSIPLS Scores'), os.path.join(DATA_DIR, 'LSIPLS Scores Train'),
                                'LSIPLS with SVC', 'r', dim_featurespace, os.path.join(DATA_DIR, 'LSIPLS Params'), PARAMETERS_SVC)

    axes[0].set_ylabel('F1 score for test data')
    axes[1].set_ylabel('F1 score for training data')
    axes[1].legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.2), fancybox=True, shadow=True, ncol=5)
    for i in range(2):
        axes[i].set_xlabel('Dimension of the projection room (*' +
                           str(LSI_DIMENSION_MULTIPLIER) + ' for LSI)', labelpad=1)
        axes[i].set_xticks(dim_featurespace)
    plt.savefig("results.pdf", bbox_inches='tight')
    plt.show()
