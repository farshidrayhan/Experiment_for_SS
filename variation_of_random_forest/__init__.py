from collections import OrderedDict

import sklearn

import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# file_obj = open('/home/farshid/Desktop/false positive.txt', 'w')
# from hsa_to_20_matrix.ada import X_train

dataset = '/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/nr_dataset_1477.txt'
df = pd.read_csv(dataset, header=None)
# df = pd.read_csv('I:/dataset/' + dataset, header=None)
# print('reading', dataset)
df['label'] = df[df.shape[1] - 1]
#
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])
sampler = RandomUnderSampler()
normalization_object = Normalizer()
X = normalization_object.fit_transform(X)
number_of_split = 5
skf = StratifiedKFold(n_splits=number_of_split, shuffle=True )

sampler = RandomUnderSampler()
# clf = DecisionTreeClassifier()
clf_list = []
estimator = []
feature_division = [1282,1294,1260,1260+300,1477,1404]

for depth in np.arange(2,120,2.5):
    for split in np.arange(2,15,2):

        avg_roc_score = 0
        avg_aupr_score = 0
        # training process
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            X_train , y_train = sampler.fit_sample(X_train,y_train)

            X_train_reduced_feature_list = []
            X_test_reduced_feature_list = []

            predicted_list = []

            predicted_array = np.empty(shape=[len(y_test),2])

            number_of_estimator = 0
            for feature in feature_division:

                number_of_estimator += 1

                clf = DecisionTreeClassifier(max_depth=depth,min_samples_split=split)
                # clf = ExtraTreeClassifier(max_depth=depth,min_samples_split=split)

                X_train_reduced = []
                X_test_reduced = []

                for i in X_train:
                    X_train_reduced.append(i[0:feature])

                X_train_reduced_feature_list.append(X_train_reduced)
                for i in X_test:
                    X_test_reduced.append(i[0:feature])
                #
                X_test_reduced_feature_list.append(X_test_reduced)

                clf.fit(X_train_reduced, y_train)
                clf_list.append(clf)

                # print(clf.predict_proba([X_test_reduced[0]]))

                predicted_list.append(clf.predict_proba(X_test_reduced))
                predicted_array += clf.predict_proba(X_test_reduced)

            # print(predicted_array)
            # predicted_array = predicted_array/number_of_estimator
            # print("roc :- ", roc_auc_score(y_test, predicted_array[:, 1]))
            avg_roc_score += roc_auc_score(y_test, predicted_array[:, 1])
            avg_aupr_score += average_precision_score(y_test, predicted_array[:, 1])
        print('for depth ', depth , 'for split ', split  , ' roc avg score - ' , avg_roc_score/number_of_split , ' roc aupr score - ' , avg_aupr_score/number_of_split )
        # avg_roc += roc_auc_score(y_test, predictions[:, 1])
        # print(avg_roc / number_of_split)
