import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from get_scores import get_score_vector
import pickle

dim_featurespace=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
fig, axes = plt.subplots(2,1,figsize=(8, 6))

#with open("Dimension Featurespace","rb") as dim_featurespace_file:
 #   stored_dim_featurespace=pickle.load(dim_featurespace_file)
#
#with open("Dimension Featurespace","wb") as dim_featurespace_file_new:
#    pickle.dump(dim_featurespace,dim_featurespace_file_new)

with open("Score SVC","rb") as svc_score_file:
    svc_best_score=pickle.load(svc_score_file)
svc_score=svc_best_score*np.ones(len(dim_featurespace))
axes[0].plot(dim_featurespace,svc_score,label='SVC',color='y')
with open("Score SVC Train","rb") as svc_score_file_train:
    svc_best_score_train=pickle.load(svc_score_file_train)
svc_score_train=svc_best_score_train*np.ones(len(dim_featurespace))
axes[1].plot(dim_featurespace,svc_score_train,label='SVC',color='y')

with open("LSI Scores","rb") as LSI_score_file:
    LSI_score=pickle.load(LSI_score_file)
with open("LSI Scores Train","rb") as LSI_score_file_train:
    LSI_score_train=pickle.load(LSI_score_file_train)
axes[0].plot(dim_featurespace,LSI_score,label='LSI mit SVC',color='b')
axes[1].plot(dim_featurespace,LSI_score_train,label='LSI mit SVC',color='b')

with open("LSR Scores","rb") as LSR_score_file:
    LSR_score=pickle.load(LSR_score_file)
with open("LSR Scores Train","rb") as LSR_score_file_train:
    LSR_score_train=pickle.load(LSR_score_file_train)

axes[0].plot(dim_featurespace,LSR_score,color='g',label='SIPLS mit Klassifizierung nach (1.6)')
axes[1].plot(dim_featurespace,LSR_score_train,label='SIPLS mit Klassifizierung nach (1.6)',color='g')

with open("LSIPLS Scores","rb") as LSIPLS_score_file:
    LSIPLS_score=pickle.load(LSIPLS_score_file)
with open("LSIPLS Scores Train","rb") as LSIPLS_score_file_train:
    LSIPLS_score_train=pickle.load(LSIPLS_score_file_train)
axes[0].plot(dim_featurespace,LSIPLS_score,label='LSIPLS mit SVC',color='r')
axes[1].plot(dim_featurespace,LSIPLS_score_train,label='LSIPLS mit SVC',color='r')

axes[0].set_ylabel('F1 Score der Testsdaten')
axes[1].set_ylabel('F1 Score der Trainingsdaten')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=5)
for i in range(2):
    axes[i].set_xlabel('Dimension des Projektionsraums (*50 für LSI)',labelpad=1)
    axes[i].set_xticks(dim_featurespace)
    
plt.savefig("ergebnisse_train and test.pdf",bbox_inches='tight')
plt.show()