# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:46:18 2017

@author: Farshid
"""
from sklearn.cluster import KMeans

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc,precision_recall_curve
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

from sklearn.metrics import roc_auc_score,average_precision_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
 
datasets = ['nr_dataset_1282.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt']

# dataset = 'gpcr_dataset_1282.txt'
            
def create_model(dataset):

    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/Desktop/'+dataset, header=None)
    
    print('reading' , dataset)
    df['label'] = df[df.shape[1] - 1]
    #
    df.drop([df.shape[1] - 2], axis=1, inplace=True)
    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])
    #
    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])

    number_of_clusters = 23
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
    print('training' , dataset)
    top_roc = 0
    
    depth_for_rus = 0
    split_for_rus = 0
    
    for depth in range(3, 20,20):
        for split in range(3, 9,20):
    
            
    
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

                
                tpr = dict()
                fpr = dict()
                roc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test,
                                                        predictions[:, i])
                    roc[i] = roc_auc_score(y_test, predictions[:, i])

    major_class = max(sampler.fit(X_train, y_train).stats_c_, key=sampler.fit(X_train, y_train).stats_c_.get)

    major_class_X_train = []
    major_class_y_train = []
    minor_class_X_train = []
    minor_class_y_train = []

    for index in range(len(X_train)):
        if y_train[index] == major_class:
            major_class_X_train.append(X_train[index])
            major_class_y_train.append(y_train[index])
        else:
            minor_class_X_train.append(X_train[index])
            minor_class_y_train.append(y_train[index])

    # optimize for number of clusters here
    kmeans = KMeans(max_iter=200, n_jobs=4, n_clusters=number_of_clusters)
    kmeans.fit(major_class_X_train)

    # get the centroids of each of the clusters
    cluster_centroids = kmeans.cluster_centers_

    # get the points under each cluster
    points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    for i in range(number_of_clusters):
        size = len(points_under_each_cluster[i])
        random_indexes = np.random.randint(low=0, high=size, size=int(size / 2))
        temp = points_under_each_cluster[i]
        feature_indexes = temp[random_indexes]
        X_train_major = np.concatenate((X_train_major, X_train[feature_indexes]), axis=0)
        y_train_major = np.concatenate((y_train_major, y_train[feature_indexes]), axis=0)

    final_train_x = np.concatenate((X_train_major, minor_class_X_train), axis=0)
    final_train_y = np.concatenate((y_train_major, minor_class_y_train), axis=0)

    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=150))
    # classifier = sklearn.svm.SVC(C=50 , gamma= .0008 , kernel='rbf', probability=True)
    # classifier = sklearn.svm.SVC(C=100, gamma=.006, kernel='rbf', probability=True)

    classifier.fit(final_train_x, final_train_y)

    predicted = classifier.predict_proba(X_test)

    tpr_c = dict()
    fpr_c = dict()
    roc_c = dict()
    for i in range(n_classes):
        fpr_c[i], tpr_c[i], _ = roc_curve(y_test,predictions[:, i])
        roc_c[i] = auc(y_test, predictions[:, i])
    
    
                
    print('ploting' , dataset)                
#    plt.clf()
    plt.plot(fpr[1], tpr[1], lw=2, color='red',
         label='Roc curve: Clustered sampling')
    
    plt.plot(fpr_c[1], tpr_c[1], lw=2, color='navy',
         label='Roc curve: random under sampling')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Area under ROC curve')
    plt.legend(loc="lower right")
    plt.show()

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




































