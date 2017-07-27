# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:46:18 2017

@author: Farshid
"""
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier

print(__doc__)

import numpy as np
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib

from itertools import cycle
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
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

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# datasets = ['enzyme_dataset_1282.txt', 'enzyme_dataset_1404.txt', ]
# datasets = ['nr_dataset_1294.txt', 'nr_dataset_1282.txt' ]

datasets = [
    'new-thyroid1.txt',
    'segment0.txt',
    'pima.txt',
    'yeast4.txt',
    'yeast5.txt',
    'yeast6.txt',
    'glass5.txt',
    'glass6.txt',

    'newthyroid2.txt',
    'yeast-2_vs_4.txt',
    'glass-0-1-2-3_vs_4-5-6.txt',
    'page-blocks-1-3_vs_4.txt'

]


# , 'nr_dataset_1294.txt', 'nr_dataset_1282.txt',
#           'gpcr_dataset_1477.txt', 'gpcr_dataset_1404.txt', 'gpcr_dataset_1294.txt', 'gpcr_dataset_1282.txt','enzyme_dataset_1282.txt']


def create_model(dataset):
    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/' + dataset, header=None)

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

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        break
    print('training', dataset)
    top_roc = 0

    depth_for_rus = 0
    split_for_rus = 0

    for depth in range(14, 20, 200):
        for split in range(8, 9, 200):

            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=depth, min_samples_split=split),
                n_estimators=100,
                learning_rate=1, algorithm='SAMME')

            X_train, y_train = sampler.fit_sample(X_train, y_train)

            classifier.fit(X_train, y_train)

            predictions = classifier.predict_proba(X_test)

            score = roc_auc_score(y_test, predictions[:, 1])

            if top_roc < score:
                top_roc = score

                depth_for_rus = depth
                split_for_rus = split

                tpr = dict()
                fpr = dict()
                roc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test,
                                                  predictions[:, i])
                    roc[i] = roc_auc_score(y_test, predictions[:, i])
                precision_c = dict()
                recall_c = dict()
                average_precision_c = dict()

                for i in range(n_classes):
                    precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
                                                                            predictions[:, i])
                    average_precision_c[i] = average_precision_score(y_test, predictions[:, i])

#########################################################
    classifier = DecisionTreeClassifier()

    # X_train, y_train = sampler.fit_sample(X_train, y_train)

    classifier.fit(X_train, y_train)
    #
    predictions = classifier.predict_proba(X_test)

    score = roc_auc_score(y_test, predictions[:, 1])

    tpr_r = dict()
    fpr_r = dict()
    roc_r = dict()
    for i in range(n_classes):
        fpr_r[i], tpr_r[i], _ = roc_curve(y_test, predictions[:, i])
        # roc_r[i] = auc(y_test, predictions[:, i])

    precision_r = dict()
    recall_r = dict()
    average_precision_r = dict()

    for i in range(n_classes):
        precision_r[i], recall_r[i], _ = precision_recall_curve(y_test,
                                                                predictions[:, i])
        # average_precision_r[i] = average_precision_score(y_test, predictions[:, i])
###################################################################
    classifier = ExtraTreeClassifier()

    # X_train, y_train = sampler.fit_sample(X_train, y_train)

    classifier.fit(X_train, y_train)
    #
    predictions = classifier.predict_proba(X_test)

    score = roc_auc_score(y_test, predictions[:, 1])

    tpr_e = dict()
    fpr_e = dict()
    roc_e = dict()
    for i in range(n_classes):
        fpr_e[i], tpr_e[i], _ = roc_curve(y_test, predictions[:, i])
        # roc_r[i] = auc(y_test, predictions[:, i])

    precision_e = dict()
    recall_e = dict()
    average_precision_e = dict()

    for i in range(n_classes):
        precision_e[i], recall_e[i], _ = precision_recall_curve(y_test,
                                                                predictions[:, i])

###################################################################

    classifier = DecisionTreeClassifier()

    # X_train, y_train = sampler.fit_sample(X_train, y_train)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict_proba(X_test)

    score = roc_auc_score(y_test, predictions[:, 1])

    tpr_s = dict()
    fpr_s = dict()
    roc_s = dict()
    for i in range(n_classes):
        fpr_s[i], tpr_s[i], _ = roc_curve(y_test, predictions[:, i])
        # roc_s[i] = auc(y_test, predictions[:, i])

    precision_s = dict()
    recall_s = dict()
    average_precision_s = dict()

    for i in range(n_classes):
        precision_s[i], recall_s[i], _ = precision_recall_curve(y_test,
                                                                predictions[:, i])
    #     average_precision_s[i] = average_precision_score(y_test, predictions[:, i])
###########################################################################

    print('ploting', dataset)
    plt.clf()
    plt.plot(fpr_s[1], tpr_s[1], lw=2, color='red', label='Roc curve: RUSboost')
    plt.plot(fpr_r[1], tpr_r[1], lw=2, color='black', label='Roc curve: SMOTEboost ')
    plt.plot(fpr_e[1], tpr_e[1], lw=2, color='green', label='Roc curve: adaboost ')
    plt.plot(fpr[1], tpr[1], lw=2, color='navy', label='Roc curve: CUSBoost')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Area under ROC curve')
    plt.legend(loc="lower right")
    plt.savefig('/home/farshid/Desktop/roc/' + dataset + '.png')
    # plt.show()

    plt.clf()
    plt.plot(recall_s[1], precision_s[1], lw=2, color='red',label='Precision-Recall RUSBoost')
    plt.plot(recall_r[1], precision_r[1], lw=2, color='black',label='Precision-Recall SMOTEBoost')
    plt.plot(recall_e[1], precision_e[1], lw=2, color='green',label='Precision-Recall adaBoost')

    plt.plot(recall_c[1], precision_c[1], lw=2, color='navy',label='Precision-Recall CUSBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall'.format(average_precision_c))
    plt.legend(loc="upper right")
    plt.savefig('/home/farshid/Desktop/aupr/' + dataset + '.png')
    # plt.show()


def func(x):
    #    print('process id:', os.getpid() )


    # math.factorial(x)
    create_model(x)


if __name__ == '__main__':
    input('Enter any Key to start ')
    pool = Pool(1)
    #    datasets= ['gpcr_dataset_1477.txt','gpcr_dataset_1404.txt']
    results = pool.map(func, datasets)

    input('Enter any Key to end ')
