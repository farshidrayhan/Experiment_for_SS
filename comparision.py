# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:46:18 2017

@author: Farshid
"""
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

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

datasets = ['gpcr_dataset_1282.txt','gpcr_dataset_1294.txt','gpcr_dataset_1404.txt','gpcr_dataset_1477.txt',
            'nr_dataset_1282.txt','nr_dataset_1294.txt','nr_dataset_1404.txt','nr_dataset_1477.txt',
            'ic_dataset_1282.txt','ic_dataset_1294.txt','ic_dataset_1404.txt','ic_dataset_1477.txt']


datasets = ['enzyme_dataset_1282.txt']
           


def create_model(dataset):
    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/Desktop/' +  dataset, header=None)
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

    for train_index, test_index in skf.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        break
    # print('training', dataset)
    X_train, y_train = sampler.fit_sample(X_train, y_train)

    top_roc = 0

    depth_for_rus = 0
    split_for_rus = 0

    for depth in range(3, 12, 100):
        for split in range(3, 9, 100):

            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=15),
                n_estimators=100,
                learning_rate=1, algorithm='SAMME')

            classifier.fit(X_train, y_train)

            predictions = classifier.predict_proba(X_test)

            score = roc_auc_score(y_test, predictions[:, 1])

            if top_roc < score:
                top_roc = score

                # print("ada score ", score , " depth " , depth , ' split ' , split)
                depth_for_rus = depth
                split_for_rus = split

                precision = dict()
                recall = dict()
                average_precision = dict()
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(y_test,
                                                                        predictions[:, i])
                    average_precision[i] = average_precision_score(y_test, predictions[:, i])
                tpr = dict()
                fpr = dict()
                roc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test,
                                                  predictions[:, i])
                    roc[i] = roc_auc_score(y_test, predictions[:, i])
    top_ada_roc = top_roc
    print(top_ada_roc)
    top_roc = 0

    for C in range(100, 1000, 100000):
        for gamma in np.arange(.009, 1, 1):
            classifier = classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=12),
                n_estimators=100,
                learning_rate=1, algorithm='SAMME')

            # X_train, y_train = sampler.fit_sample(X_train, y_train)

            classifier.fit(X_train, y_train)

            predictions = classifier.predict_proba(X_test)

            score = roc_auc_score(y_test, predictions[:, 1])

            if top_roc < score:
                top_roc = score

                # print("svm score ", score ,  " C " , C , ' split ' , gamma)

                precision_c = dict()
                recall_c = dict()
                average_precision_c = dict()

                for i in range(n_classes):
                    precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
                                                                            predictions[:, i])
                    average_precision_c[i] = average_precision_score(y_test, predictions[:, i])

                tpr_c = dict()
                fpr_c = dict()
                roc_c = dict()
                for i in range(n_classes):
                    fpr_c[i], tpr_c[i], _ = roc_curve(y_test, predictions[:, i])
                    roc_c[i] = roc_auc_score(y_test, predictions[:, i])
    top_svm_roc = top_roc
    print(top_svm_roc)
    top_roc = 0
    for depth in range(2, 20, 100):
        for split in np.arange(2, 9, 100):
            classifier = RandomForestClassifier(n_estimators=100, max_depth=15 , min_samples_split= 8)

            # X_train, y_train = sampler.fit_sample(X_train, y_train)


            classifier.fit(X_train, y_train)

            predictions = classifier.predict_proba(X_test)

            score = roc_auc_score(y_test, predictions[:, 1])

            if top_roc < score:
                top_roc = score

                # print("RF score ", score, " depth " , depth , ' split ' , split)

                precision_r = dict()
                recall_r = dict()
                average_precision_r = dict()

                for i in range(n_classes):
                    precision_r[i], recall_r[i], _ = precision_recall_curve(y_test,
                                                                            predictions[:, i])
                    average_precision_r[i] = average_precision_score(y_test, predictions[:, i])

                tpr_r = dict()
                fpr_r = dict()
                roc_r = dict()
                for i in range(n_classes):
                    fpr_r[i], tpr_r[i], _ = roc_curve(y_test, predictions[:, i])
                    roc_r[i] = roc_auc_score(y_test, predictions[:, i])

    top_rf_roc = top_roc
    print(top_rf_roc)

    print('ploting', dataset , "\n svm roc " , top_svm_roc , ' aupr ' , average_precision_c[0] + average_precision_c[1] ,
" ada roc " , top_ada_roc , ' aupr ' , average_precision[0] + average_precision[1] ,
" RF roc " , top_rf_roc , ' aupr ' , average_precision_r[0] + average_precision_r[1])
    #    plt.clf()
    plt.plot(recall_c[1], precision_c[1], lw=2, color='red', label='Precision-Recall SVM')

    plt.plot(recall_r[1], precision_r[1], lw=2, color='green', label='Precision-Recall Random Forest ')

    plt.plot(recall[1], precision[1], lw=2, color='navy',label='Precision-Recall adaBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall '.format(average_precision))
    plt.legend(loc="upper right")
    # plt.savefig('/home/uiu/protein_thesis/aupr/' + dataset + '.png')
    plt.show()
    plt.close()

    plt.plot(fpr_c[1], tpr_c[1], lw=2, color='red', label='Roc curve: SVM')
    plt.plot(fpr_r[1], tpr_r[1], lw=2, color='green', label='Roc curve: Random Forest ')
    plt.plot(fpr[1], tpr[1], lw=2, color='navy',label='Roc curve: adaBoost')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Area under ROC curve')
    plt.legend(loc="lower right")
    # plt.savefig('/home/uiu/protein_thesis/roc/' + dataset + '.png')
    plt.show()
    plt.close()


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
