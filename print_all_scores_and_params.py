import pickle

import numpy as np

from print_results import print_test_train_params

with open("Score SVC", "rb") as svc_score_file:
    svc_best_score = pickle.load(svc_score_file)
with open("Score SVC Train", "rb") as svc_score_file_train:
    svc_best_score_train = pickle.load(svc_score_file_train)
with open("SVC best params", "rb") as svc_params_file:
    svc_best_params = pickle.load(svc_params_file)

print('___SVC___')
print('Testscore   Trainingsscore   beste Parameter')
print(np.round(svc_best_score, 4), np.round(svc_best_score_train, 4), svc_best_params)

with open("LSI Scores", "rb") as LSI_score_file:
    LSI_score = pickle.load(LSI_score_file)
with open("LSI Scores Train", "rb") as LSI_score_file_train:
    LSI_score_train = pickle.load(LSI_score_file_train)
with open("LSI Params", "rb") as LSI_params_file:
    LSI_params = pickle.load(LSI_params_file)

with open("LSIPLS Scores", "rb") as LSIPLS_score_file:
    LSIPLS_score = pickle.load(LSIPLS_score_file)
with open("LSIPLS Scores Train", "rb") as LSIPLS_score_file_train:
    LSIPLS_score_train = pickle.load(LSIPLS_score_file_train)
with open("LSIPLS Params", "rb") as LSIPLS_params_file:
    LSIPLS_params = pickle.load(LSIPLS_params_file)

with open("LSR Scores", "rb") as LSR_score_file:
    LSR_score = pickle.load(LSR_score_file)
with open("LSR Scores Train", "rb") as LSR_score_file_train:
    LSR_score_train = pickle.load(LSR_score_file_train)

print('____LSI___')
print_test_train_params(LSI_score, LSI_score_train, LSI_params, range(50, 800, 50))

print('___SIPLS___')
print_test_train_params(LSR_score, LSR_score_train)

print('___LSIPLS___')
print_test_train_params(LSIPLS_score, LSIPLS_score_train, LSIPLS_params)
