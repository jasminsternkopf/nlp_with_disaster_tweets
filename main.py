import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from get_scores import get_score_vector
import pickle

#warnings.filterwarnings("ignore")

if __name__ == "__main__":

    dim_featurespace=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    fig, ax = plt.subplots()

    with open("Dimension Featurespace","rb") as dim_featurespace_file:
        stored_dim_featurespace=pickle.load(dim_featurespace_file)

    with open("Dimension Featurespace","wb") as dim_featurespace_file_new:
        pickle.dump(dim_featurespace,dim_featurespace_file_new)

    parameters_SVC={'Classifier__C':(0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100),'Classifier__kernel':('linear','rbf','poly','sigmoid')}

    with open("Score SVC","rb") as svc_score_file:
        svc_best_score=pickle.load(svc_score_file)
    svc_score=svc_best_score*np.ones(len(dim_featurespace))
    ax.plot(dim_featurespace,svc_score,label='SVC',color='y')
    with open("Score SVC Train","rb") as svc_score_file_train:
        svc_best_score_train=pickle.load(svc_score_file_train)
    svc_score_train=svc_best_score_train*np.ones(len(dim_featurespace))
    ax.plot(dim_featurespace,svc_score_train,linestyle='--',color='y')

    with open("LSI Scores","rb") as LSI_score_file:
        LSI_score=pickle.load(LSI_score_file)
    with open("LSI Scores Train","rb") as LSI_score_file_train:
        LSI_score_train=pickle.load(LSI_score_file_train)
    if stored_dim_featurespace==dim_featurespace and len(dim_featurespace)==len(LSI_score) and False:
        ax.plot(dim_featurespace,LSI_score,label='LSI mit SVC',color='b')
        ax.plot(dim_featurespace,LSI_score_train,linestyle='--',color='b')
        print("LSI scores were already computed")
    else:
        LSI_score,LSI_score_train,LSI_params=get_score_vector('LSI',50*np.array(dim_featurespace),parameters_SVC)
        ax.plot(dim_featurespace,LSI_score,label='LSI mit SVC',color='b')
        ax.plot(dim_featurespace,LSI_score_train,linestyle='--',color='b')#,label='LSI mit SVC - Trainingsdaten')
        with open("LSI Scores","wb") as LSI_score_file:
            pickle.dump(LSI_score,LSI_score_file)
        with open("LSI Scores Train","wb") as LSI_score_file_train:
            pickle.dump(LSI_score_train,LSI_score_file_train)
        with open("LSI Params","wb") as LSI_params_file:
            pickle.dump(LSI_params,LSI_params_file)

    with open("LSR Scores","rb") as LSR_score_file:
        LSR_score=pickle.load(LSR_score_file)
    with open("LSR Scores Train","rb") as LSR_score_file_train:
        LSR_score_train=pickle.load(LSR_score_file_train)
    if stored_dim_featurespace==dim_featurespace and len(dim_featurespace)==len(LSR_score):
        ax.plot(dim_featurespace,LSR_score,color='g',label='SIPLS mit eigener Klassifizierung')
        ax.plot(dim_featurespace,LSR_score_train,linestyle='--',color='g')
        print("LSR scores were already computed")
    else:
        LSR_score,LSR_score_train=get_score_vector('LSR_Estimator',dim_featurespace)
        ax.plot(dim_featurespace,LSR_score,color='g',label='SIPLS mit eigener Klassifizierung')
        ax.plot(dim_featurespace,LSR_score_train,linestyle='--',color='g')
        with open("LSR Scores","wb") as LSR_score_file:
            pickle.dump(LSR_score,LSR_score_file)
        with open("LSR Scores Train","wb") as LSR_score_file_train:
            pickle.dump(LSR_score_train,LSR_score_file_train)

    with open("LSIPLS Scores","rb") as LSIPLS_score_file:
        LSIPLS_score=pickle.load(LSIPLS_score_file)
    with open("LSIPLS Scores Train","rb") as LSIPLS_score_file_train:
        LSIPLS_score_train=pickle.load(LSIPLS_score_file_train)
    if stored_dim_featurespace==dim_featurespace and len(dim_featurespace)==len(LSIPLS_score):
        ax.plot(dim_featurespace,LSIPLS_score,label='LSIPLS mit SVC',color='r')
        ax.plot(dim_featurespace,LSIPLS_score_train,linestyle='--',color='r')
        print("LSIPLS scores were already computed")
    else:
        LSIPLS_score,LSIPLS_score_train,LSIPLS_params=get_score_vector('LSIPLS',dim_featurespace,parameters_SVC)
        ax.plot(dim_featurespace,LSIPLS_score,label='LSIPLS mit SVC',color='r')
        ax.plot(dim_featurespace,LSIPLS_score_train,linestyle='--',color='r')
        with open("LSIPLS Scores","wb") as LSIPLS_score_file:
            pickle.dump(LSIPLS_score,LSIPLS_score_file)
        with open("LSIPLS Params","wb") as LSIPLS_params_file:
            pickle.dump(LSIPLS_params,LSIPLS_params_file)
        with open("LSIPLS Scores Train","wb") as LSIPLS_score_file_train:
            pickle.dump(LSIPLS_score_train,LSIPLS_score_file_train)

    
    ax.set_xlabel('Dimension des Unterraums (*100 für LSI)')
    ax.set_ylabel('F1 score')
    ax.set_xticks(dim_featurespace)
    ax.legend()
    plt.savefig("ergebnisse.pdf",bbox_inches='tight')
    plt.show()