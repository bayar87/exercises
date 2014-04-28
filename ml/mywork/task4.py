"""
Task 4:

In this task we present a real life dataset in the form of a supervised classification problem.
Ref: data/coupon_clickstream.csv

This dataset contains 50 observations and one target variable.
What we are trying to predict here is that given these 50 metrics how likely is a user to click on a coupon.

Your task is the following:

Perform exploratory analysis on the dataset to answer following questions:

1. Are there any redundant metrics in the 50 variables we have collected?
2. Are there any correlated metrics?
3. Will dimensionality reduction need to be applied?

Once you know what you are looking at perform the following tasks:

1. Find the optimal number of features that maximize the accuracy of predicting 'coupon_click'
2. Once you identify optimal number of features can you rank the features from most important to least?

Use optimal features you found above with different classifiers and evaulate your classifiers as to how general they are with relevant metrics.

"""
######################################Classification & Feature Selection ############################################

import numpy
import numpy as np
import scipy
from scipy.stats import *



data = numpy.genfromtxt('coupon_clickstream.csv', delimiter = ',')
nbr_Samp=data.shape[0]-1
nbr_Ftrs=data.shape[1]-1
metrics=data[1:,:nbr_Ftrs]
labels=data[1:,nbr_Ftrs]


# function to remove duplicate columns (metrics)
def duplicate_columns(data, minoccur=2):
    ind = np.lexsort(data)
    diff = np.any(data.T[ind[1:]] != data.T[ind[:-1]], axis=1)
    edges = np.where(diff)[0] + 1
    result = np.split(ind, edges)
    result = [group for group in result if len(group) >= minoccur]
    return result

print "The duplicate metrics are %s" % duplicate_columns(metrics)

#remove columns
metrics_unique = np.delete(metrics, np.s_[24,28,30,40,46], 1)

nbr_Ftrs_uq = metrics_unique.shape[1] 
#Generate a correlation matrix
dimCorr  = (nbr_Ftrs_uq, nbr_Ftrs_uq)
#initialize the correlation matrix to zero entries
r  = np.zeros(dimCorr)

#initilize the number of correlated features (columns)
nbr_Corr_Ftrs = 0



#We consider two features are highly correlated if their correlation coefficient is bigger than 0.4
for i in range(nbr_Ftrs_uq): # rows are the number of rows in the matrix. 
    for j in range(i, nbr_Ftrs_uq):
        d= scipy.stats.pearsonr(metrics_unique[:,i], metrics_unique[:,j])
        r[i,j] = d[0]
        r[j,i] = r[j,i]
        if i!=j and r[i,j] > 0.4 and r[i,j] not in (0,1):
        	nbr_Corr_Ftrs +=1
        	print "The feautures %d and %d are highly correlated and their correlation coefficient is equal to %0.3f" % (i,j,r[i,j])

print "There are only %d highly correlated features with correlation coefficient bigger than 0.4" % nbr_Corr_Ftrs
