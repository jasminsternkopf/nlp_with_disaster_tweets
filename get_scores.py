from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from get_data import get_data
from LSR_Classes import LSR_Estimator, LSR_Transformer, LSIPLS, Centerer, Row_Normalizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
import multiprocessing
from typing import Optional, Tuple, Dict

def get_score_vector(Transformer,dim_featurespace,parameters=None):
    all_scores=np.zeros(len(dim_featurespace))
    all_scores_train=np.zeros(len(dim_featurespace))
    best_parameters=[]
    classifier=SVC()
    use_cpus=multiprocessing.cpu_count()-1
    if Transformer=='LSR_Transformer':
        for i,no_of_features in enumerate(dim_featurespace):
            X_train,y_train,X_test,y_test=get_data(3)
            pipe=Pipeline([('Center',Centerer()),('Transformer',LSR_Transformer(no_of_features)),('Normalizer',Row_Normalizer()),('Classifier',classifier)])
            pipe.fit(X_train,y_train)
            y_pred=pipe.predict(X_test)
            all_scores[i]=f1_score(y_pred,y_test)
            y_pred_train=pipe.predict(X_train)
            all_scores_train[i]=f1_score(y_pred_train,y_train)
            print("LSR erledigt für Featureanzahl=",no_of_features)
    elif Transformer=='LSIPLS':
        for i,no_of_features in enumerate(tqdm(dim_featurespace)):
            X_train,y_train,X_test,y_test=get_data(3)
            pipe=Pipeline([('Center',Centerer()),('Transformer',LSIPLS(no_of_features)),('Normalizer',Row_Normalizer()),('Classifier',classifier)])
            grid_search=GridSearchCV(pipe,parameters,scoring='f1',n_jobs=use_cpus)
            grid_search.fit(X_train,y_train)
            y_pred=grid_search.predict(X_test)
            all_scores[i]=f1_score(y_pred,y_test)
            y_pred_train=grid_search.predict(X_train)
            all_scores_train[i]=f1_score(y_pred_train,y_train)
            best_parameters.append(grid_search.best_params_)
            print("LSIPLS erledigt für Featureanzahl=",no_of_features)
    elif Transformer=='LSR_Estimator':
        for i,no_of_features in enumerate(dim_featurespace):
            X_train,y_train,X_test,y_test=get_data(3)
            pipe=Pipeline([('Center',Centerer()),('LSR',LSR_Estimator(no_of_features))])
            pipe.fit(X_train,y_train)
            y_pred=pipe.predict(X_test)
            all_scores[i]=f1_score(y_pred,y_test)
            y_pred_train=pipe.predict(X_train)
            all_scores_train[i]=f1_score(y_pred_train,y_train)
            print("LSR erledigt für Featureanzahl=",no_of_features)
    else: #Transformer=='LSI'
        for i,no_of_features in enumerate(tqdm(dim_featurespace)):
            X_train,y_train,X_test,y_test=get_data(3)
            pipe=Pipeline([('Center',Centerer()),('Transformer',TruncatedSVD(no_of_features,random_state=1)),('Normalizer',Row_Normalizer()),('Classifier',classifier)])
            grid_search=GridSearchCV(pipe,parameters,scoring='f1',n_jobs=use_cpus)
            grid_search.fit(X_train,y_train)
            y_pred=grid_search.predict(X_test)
            all_scores[i]=f1_score(y_pred,y_test)
            y_pred_train=grid_search.predict(X_train)
            all_scores_train[i]=f1_score(y_pred_train,y_train)
            best_parameters.append(grid_search.best_params_)
            print("LSI erledigt für Featureanzahl=",no_of_features)
    if Transformer=='LSR_Estimator':
        return all_scores,all_scores_train
    else:
        return all_scores,all_scores_train,best_parameters

