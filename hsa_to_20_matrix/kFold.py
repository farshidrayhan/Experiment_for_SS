from sklearn import cross_validation
import random


import re

from sklearn import svm, metrics
import sklearn
# from sklearn.model_selection import cross_val_predic
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np
from sklearn import datasets


clf = svm.SVC()
data = []
for i in range(0,26):
    data.append(i)

kf = cross_validation.KFold(25,n_folds=5 )

for iteration ,data in enumerate(kf,start=1):
    print(iteration,data[0],data[1])
    # clf.fit(data[0],data[1])

