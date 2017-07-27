# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:46:18 2017

@author: Farshid
"""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.externals import joblib
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
from multiprocessing import Pool, Lock
import math
from math import factorial

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

datasets = ['gpcr_dataset_1477.txt', 'gpcr_dataset_1404.txt', 'gpcr_dataset_1294.txt', 'gpcr_dataset_1282.txt',
            'ic_dataset_1282.txt', 'ic_dataset_1404.txt', 'ic_dataset_1294.txt', 'ic_dataset_1282.txt']
datasets = ['nr_dataset_1477.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt']

def create_model(dataset):
    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/Desktop/' + dataset, header=None)

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

    for depth in range(3, 20, 20):
        for split in range(3, 9, 20):

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

                print("roc score " , top_roc)
                print("aupr score ", average_precision_score(y_test, predictions[:, 1]))

                y_pred = classifier.predict(X_test)

                print("MCC : ",matthews_corrcoef(y_test,y_pred) )
                confusion_matri = confusion_matrix(y_test,y_pred)
                print(confusion_matri)
                print(classification_report(y_test,y_pred)  )

                print("specificity ", end='')
                specificity = float(confusion_matri[0][0]) / (float(confusion_matri[0][0]) + float(confusion_matri[0][1]))
                print(specificity)
                print("sensitivity ", end='')
                sensitivity = float(confusion_matri[1][1]) / float((confusion_matri[1][1]) + float(confusion_matri[1][0]))
                print(sensitivity)


                precision = dict()
                recall = dict()
                average_precision = dict()
                for i in range(n_classes):
                    precision[i], recall[i], _ = precision_recall_curve(y_test,
                                                                        predictions[:, i])
                    average_precision[i] = average_precision_score(y_test, predictions[:, i])

    # classifier = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=depth_for_rus - 1, min_samples_split=split_for_rus - 1),
    #     n_estimators=100,
    #     learning_rate=1, algorithm='SAMME')
    #
    # X_train, y_train = sampler.fit_sample(X_train, y_train)
    #
    # classifier.fit(X_train, y_train)
    #
    # predictions = classifier.predict_proba(X_test)
    #
    # score = roc_auc_score(y_test, predictions[:, 1])
    #
    # precision_c = dict()
    # recall_c = dict()
    # average_precision_c = dict()
    #
    # for i in range(n_classes):
    #     precision_c[i], recall_c[i], _ = precision_recall_curve(y_test,
    #                                                             predictions[:, i])
    #     average_precision_c[i] = average_precision_score(y_test, predictions[:, i])
    #
    # lock.acquire()
    # print('ploting', dataset)
    # plt.clf()
    # plt.plot(recall[1], precision[1], lw=2, color='red',
    #          label='Precision-Recall Clustered sampling')
    #
    # plt.plot(recall_c[1], precision_c[1], lw=2, color='navy',
    #          label='Precision-Recall random under sampling')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall'.format(average_precision))
    # plt.legend(loc="upper right")
    # plt.savefig('/home/farshid/Desktop/aupr/' + dataset + '.png')
    # # plt.show()
    # lock.release()


def func(x):
    #    print('process id:', os.getpid() )


    # math.factorial(x)
    create_model(x)


if __name__ == '__main__':
    input('Enter any Key to start ')
    lock = Lock()
    pool = Pool(1)
    #    datasets= ['gpcr_dataset_1477.txt','gpcr_dataset_1404.txt']
    results = pool.map(create_model, datasets )


    input('Enter any Key to end ')
