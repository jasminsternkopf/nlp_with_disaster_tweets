import numbers
import os

import numpy as np
import pandas as pd

from global_parameters import DATA_DIR
from plot_scores import open_pickle


def print_test_train_with_params(testfile, trainfile, filename, paramsfile=None, dims=None, round=4):
    """
    Prints the test and training score(s) with the corresponding hyperparameters of SVC (not for SIPLS) depending on the dimension of the projection room (not for SVC) and saves them as a .csv file
    """
    test = open_pickle(testfile)
    train = open_pickle(trainfile)
    if dims is None and not(isinstance(test, numbers.Number)):  # intended for SIPLS and LSIPLS
        dims = range(1, len(test) + 1)
    best_index = np.argmax(test)
    if paramsfile is None:  # case SIPLS
        dim_and_scores = {
            'Dimension': dims,
            'Test score': np.round(test, round),
            'Training score': np.round(train, round)
        }
        dim_and_scores_df = pd.DataFrame(dim_and_scores, columns=[
                                         'Dimension', 'Test score', 'Training score'])
        print(dim_and_scores_df.to_string(index=False))
        print(f'Best test score is {test[best_index]}, reached with dimension {dims[best_index]}.')
        dim_and_scores_df.to_csv(filename, index=False)
    else:
        params = open_pickle(paramsfile)
        if dims is None:  # case SVC
            scores_and_params = {
                'Test score': [np.round(test, round)],
                'Training score': [np.round(train, round)],
                'C': [params['Classifier__C']],
                'Kernel': [params['Classifier__kernel']]
            }
            scores_and_params_df = pd.DataFrame(scores_and_params, columns=[
                'Test score', 'Training score', 'C', 'Kernel'])
            print(scores_and_params_df.to_string(index=False))
            scores_and_params_df.to_csv(filename, index=False)
        else:  # case LSI and LSIPLS
            dim_and_scores = {
                'Dimension': dims,
                'Test score': np.round(test, round),
                'Training score': np.round(train, round),
                'C': [p['Classifier__C'] for p in params],
                'Kernel': [p['Classifier__kernel'] for p in params]
            }
            dim_and_scores_df = pd.DataFrame(dim_and_scores, columns=[
                'Dimension', 'Test score', 'Training score', 'C', 'Kernel'])
            print(dim_and_scores_df.to_string(index=False))
            print(f"Best test score is {test[best_index]}, reached with dimension {dims[best_index]}, regularization parameter C = {params[best_index]['Classifier__C']} and kernel {params[best_index]['Classifier__kernel']}.")
            dim_and_scores_df.to_csv(filename, index=False)


def print_all_scores_with_params():
    """
    Prints the scores (with the corresponding parameters and dimension) of all models, i.e. SVC, LSI, SIPLS and LSIPLS
    """
    print('SVC:\n')
    print_test_train_with_params(os.path.join(DATA_DIR, 'SVC Score'), os.path.join(DATA_DIR, 'SVC Score Train'), os.path.join(DATA_DIR, 'svc_scores.csv'),
                                 os.path.join(DATA_DIR, 'SVC Params'))

    print('\nLSI:\n')
    print_test_train_with_params(os.path.join(DATA_DIR, 'LSI Scores'), os.path.join(DATA_DIR, 'LSI Scores Train'), os.path.join(DATA_DIR, 'lsi_scores.csv'),
                                 os.path.join(DATA_DIR, 'LSI Params'), range(50, 800, 50))

    print('\nSIPLS:\n')
    print_test_train_with_params(os.path.join(DATA_DIR, 'SIPLS Scores'), os.path.join(
        DATA_DIR, 'SIPLS Scores Train'), os.path.join(DATA_DIR, 'sipls_scores.csv'))

    print('\nLSIPLS:\n')
    print_test_train_with_params(os.path.join(DATA_DIR, 'LSIPLS Scores'), os.path.join(
        DATA_DIR, 'LSIPLS Scores Train'), os.path.join(DATA_DIR, 'lsipls_scores.csv'), os.path.join(DATA_DIR, 'LSIPLS Params'))
