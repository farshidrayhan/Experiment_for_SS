# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:46:18 2017

@author: Farshid
"""
from collections import OrderedDict

import sklearn

print(__doc__)

import numpy as np
from itertools import cycle
from sklearn.externals import joblib
from sklearn.ensemble import *
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from multiprocessing import Pool
import math
from math import factorial
import matplotlib
import operator
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# datasets = [ 'gpcr_dataset_1477.txt',
#             'nr_dataset_1282.txt', 'nr_dataset_1294.txt', 'nr_dataset_1404.txt', 'nr_dataset_1477.txt',
#             'ic_dataset_1282.txt', 'ic_dataset_1294.txt', 'ic_dataset_1404.txt', 'ic_dataset_1477.txt']

datasets = ['gpcr_dataset_1477.txt']


def create_model(dataset):
    print("dataset : ", dataset)
    file_obj = open('/home/farshid/Desktop/false positive.txt' , 'w')
    df = pd.read_csv('/home/farshid/Desktop/' + dataset, header=None)
    # df = pd.read_csv('I:/dataset/' + dataset, header=None)
    print('reading', dataset)
    df['label'] = df[df.shape[1] - 1]
    #
    df.drop([df.shape[1] - 2], axis=1, inplace=True)
    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])
    #
    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])
    sampler = RandomUnderSampler()
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    n_classes = 2

    clf_e = joblib.load('/home/farshid/Dropbox/django_site/media/gpcr.pkl')

    predict = clf_e.predict_proba(X)

    fp_counter = 0
    rank = {}
    for i in range(len(y)):
        if y[i] == 0 and predict[i][1] > .5:
            # temp = temp.reshape(1, -1)
            # proba = clf_e.predict_proba(np.array(X[i]).reshape(1, -1))
            # if  proba[:,1] > .731 :                # .731 gpcr     # .66 nr  # .635 ic
            #     fp_counter += 1
            rank[i] = predict[i][1]
            # print(i , 'proba ' , predict[i])
    # sorted_x = sorted(rank.items(), key=operator.itemgetter(1))
    # for w in rank:
    #     print(w, rank[w] )
    # print('sorted ...................................................................................')
    # print(sorted(rank.values() , key = rank.get() ))
    rank = OrderedDict(sorted(rank.items(), key=operator.itemgetter(1)) )
    for w in rank:
        strg = 'index = ' + str(w) + ' prb of FP = ' + str(float( rank[w] + .2 )) + '\n'
        print(strg ,end='')
        file_obj.write(strg)


    file_obj.close()



    file_obj.close()



def func(x):
   # print('process id:', os.getpid() )


    # math.factorial(x)
    create_model(x)


if __name__ == '__main__':
    input('Enter any Key to start ')
    pool = Pool(1)
    #    datasets= ['gpcr_dataset_1477.txt','gpcr_dataset_1404.txt']
    results = pool.map(func, datasets)

    input('Enter any Key to end ')
