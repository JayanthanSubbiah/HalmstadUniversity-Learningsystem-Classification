
# coding: utf-8

# In[3]:


import scipy.io
import numpy as np
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

import time


# In[4]:


imported_data = scipy.io.loadmat('wineITrain.mat')

# Imported data is made up CancerTrainX, CancerTrainY, and CancerTextX
# Extract out the training and testing set
wineTrainX = imported_data['wineTrainX']
wineTrainX = wineTrainX.T

wineTrainY = imported_data['wineTrainY']
wineTrainY = wineTrainY.T

wineTestX = imported_data['wineTestX']
wineTestX = wineTestX.T


# In[5]:


def fisher_index_calc(trainingSet, labelSet):
    (dim1_T, dim2_T) = trainingSet.shape
    (dim1_L, dim2_L) = labelSet.shape

    
    fisher_ratios = np.zeros((1, dim2_T), dtype=float).flatten()
   
    if dim1_L != dim1_T:
        return fisher_ratios

    #take out the number of features avaliable 
    # group the both data together
    train1 = pd.DataFrame(trainingSet)
    label1 = pd.DataFrame(labelSet, columns=['LABEL'])
    grouped = pd.concat([train1, label1], axis=1)
    # number of classes
    (no_classes, demo) = grouped.groupby('LABEL').count()[[0]].shape
    for j in range(dim2_T):
        # mean and variance of j 
        j_variance = np.var(trainingSet[:,j])
        j_mean = np.mean(trainingSet[:,j])
        j_summation = 0
        for k in range(no_classes):
            output = grouped.groupby('LABEL').count()[[j]]
            k_feature_count = output.iloc[k,0]
            output = grouped.groupby('LABEL').mean()[[j]]
            k_feature_mean = output.iloc[k,0]
            currentSum = k_feature_count * np.square((k_feature_mean - j_mean))
            j_summation = j_summation + currentSum
        fisher_ratios[j] = j_summation / np.square(j_variance)

    return fisher_ratios
fisher_ratios = fisher_index_calc(wineTrainX, wineTrainY)


# In[6]:


df = pd.DataFrame({'Fisher Ratio For All Features': fisher_ratios})
ax = df.plot.bar()
plt.show()

