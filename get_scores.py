import multiprocessing

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from get_data import get_data
from new_classes import LSIPLS, SIPLS, Centerer, Row_Normalizer


def get_score_vector(transformer, dim_featurespace, parameters=None):
    all_scores = np.zeros(len(dim_featurespace))
    all_scores_train = np.zeros(len(dim_featurespace))
    best_parameters = []
    classifier = SVC()
    use_cpus = multiprocessing.cpu_count() - 1
    if transformer == 'LSIPLS':
        for i, no_of_features in enumerate(tqdm(dim_featurespace)):
            X_train, y_train, X_test, y_test = get_data(3)
            pipe = Pipeline([('Center', Centerer()), ('Transformer', LSIPLS(
                no_of_features)), ('Normalizer', Row_Normalizer()), ('Classifier', classifier)])
            grid_search = GridSearchCV(pipe, parameters, scoring='f1', n_jobs=use_cpus)
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
            all_scores[i] = f1_score(y_pred, y_test)
            y_pred_train = grid_search.predict(X_train)
            all_scores_train[i] = f1_score(y_pred_train, y_train)
            best_parameters.append(grid_search.best_params_)
            print("LSIPLS done for number of features =", no_of_features)
    elif transformer == 'SIPLS':
        for i, no_of_features in enumerate(dim_featurespace):
            X_train, y_train, X_test, y_test = get_data(3)
            pipe = Pipeline([('Center', Centerer()), ('SIPLS', SIPLS(no_of_features))])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            all_scores[i] = f1_score(y_pred, y_test)
            y_pred_train = pipe.predict(X_train)
            all_scores_train[i] = f1_score(y_pred_train, y_train)
            print("LSR done for number of features =", no_of_features)
    elif transformer == 'LSI':  # Transformer=='LSI'
        for i, no_of_features in enumerate(tqdm(dim_featurespace)):
            X_train, y_train, X_test, y_test = get_data(3)
            pipe = Pipeline([('Center', Centerer()), ('Transformer', TruncatedSVD(
                no_of_features, random_state=1)), ('Normalizer', Row_Normalizer()), ('Classifier', classifier)])
            grid_search = GridSearchCV(pipe, parameters, scoring='f1', n_jobs=use_cpus)
            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
            all_scores[i] = f1_score(y_pred, y_test)
            y_pred_train = grid_search.predict(X_train)
            all_scores_train[i] = f1_score(y_pred_train, y_train)
            best_parameters.append(grid_search.best_params_)
            print("LSI done for number of features =", no_of_features)
    else:  # Transformer=None, the baseline scenario
        X_train, y_train, X_test, y_test = get_data(3)
        pipe = Pipeline([('Center', Centerer()), ('Classifier', classifier)])
        grid_search = GridSearchCV(pipe, parameters, scoring='f1', n_jobs=use_cpus)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        all_scores = f1_score(y_pred, y_test)
        y_pred_train = grid_search.predict(X_train)
        all_scores_train = f1_score(y_pred_train, y_train)
        best_parameters.append(grid_search.best_params_)
    if transformer == 'SIPLS':
        return all_scores, all_scores_train, None
    else:
        return all_scores, all_scores_train, best_parameters
