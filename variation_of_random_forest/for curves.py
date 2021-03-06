from collections import OrderedDict

import sklearn

import pandas as pd
import numpy as np
from sklearn.preprocessing import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

# file_obj = open('/home/farshid/Desktop/false positive.txt', 'w')
# from hsa_to_20_matrix.ada import X_train

dataset = '/home/farshid/PycharmProjects/Experiments_for_DMF/try_1/datasets/binary/gpcr_dataset_1477.txt'
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
skf = StratifiedKFold(n_splits=number_of_split, shuffle=True)

sampler = RandomUnderSampler()
# clf = DecisionTreeClassifier()
clf_list = []
estimator = []
feature_division = [1282, 1294, 1477, 1404]
top_roc = 0
for depth in np.arange(2, 50, 2.5):
    for split in np.arange(2, 15, 2):

        avg_roc_score = 0
        avg_aupr_score = 0
        # training process
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            X_train, y_train = sampler.fit_sample(X_train, y_train)

            X_train_reduced_feature_list = []
            X_test_reduced_feature_list = []

            predicted_list = []

            predicted_array = np.empty(shape=[len(y_test), 2])

            number_of_estimator = 0
            for feature in feature_division:

                number_of_estimator += 1

                clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
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
            try:
                avg_roc_score += roc_auc_score(y_test, predicted_array[:, 1])
                avg_aupr_score += average_precision_score(y_test, predicted_array[:, 1])
            except:
                print('')
            if avg_aupr_score > top_roc:
                top_roc = avg_aupr_score

            try:
                precision = dict()
                recall = dict()
                average_precision = dict()
                for i in range(2):
                    precision[i], recall[i], _ = precision_recall_curve(y_test,
                                                                        predicted_array[:, i])
            except:
                tpr = dict()
                fpr = dict()
                roc = dict()
                # for i in range(2):
                #     fpr[i], tpr[i], _ = roc_curve(y_test,
                #                                   predicted_array[:, i])

        print('for depth ', depth, 'for split ', split, ' roc avg score - ', avg_roc_score / number_of_split,
              ' roc aupr score - ', avg_aupr_score / number_of_split)

e_precision = dict()
e_recall = dict()
for depth in np.arange(2, 50, 2.5):
    for split in np.arange(2, 15, 2):

        avg_roc_score = 0
        avg_aupr_score = 0
        # training process
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            X_train, y_train = sampler.fit_sample(X_train, y_train)

            X_train_reduced_feature_list = []
            X_test_reduced_feature_list = []

            predicted_list = []

            predicted_array = np.empty(shape=[len(y_test), 2])

            number_of_estimator = 0
            for feature in feature_division:

                number_of_estimator += 1

                # clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
                clf = ExtraTreeClassifier(max_depth=depth,min_samples_split=split)

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
            try:
                avg_roc_score += roc_auc_score(y_test, predicted_array[:, 1])
                avg_aupr_score += average_precision_score(y_test, predicted_array[:, 1])
            except:
                print('')
            if avg_aupr_score > top_roc:
                top_roc = avg_aupr_score
            try:
                average_precision = dict()
                for i in range(2):
                    e_precision[i], e_recall[i], _ = precision_recall_curve(y_test,
                                                                        predicted_array[:, i])
            except:
                tpr = dict()
                fpr = dict()
                roc = dict()
                # for i in range(2):
                #     fpr[i], tpr[i], _ = roc_curve(y_test,
                #                                   predicted_array[:, i])

        print('for depth ', depth, 'for split ', split, ' roc avg score - ', avg_roc_score / number_of_split,
              ' roc aupr score - ', avg_aupr_score / number_of_split)

s_precision = dict()
s_recall = dict()
for depth in np.arange(1, 20, 1):
    for split in np.arange(.001, .01, .002):

        avg_roc_score = 0
        avg_aupr_score = 0
        # training process
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            X_train, y_train = sampler.fit_sample(X_train, y_train)

            X_train_reduced_feature_list = []
            X_test_reduced_feature_list = []

            predicted_list = []

            predicted_array = np.empty(shape=[len(y_test), 2])

            number_of_estimator = 0
            for feature in feature_division:

                number_of_estimator += 1

                # clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
                clf = sklearn.svm.SVC(C=depth,gamma=split,probability=True)

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
            try:
                avg_roc_score += roc_auc_score(y_test, predicted_array[:, 1])
                avg_aupr_score += average_precision_score(y_test, predicted_array[:, 1])
            except:
                print('')
            if avg_aupr_score > top_roc:
                top_roc = avg_aupr_score
            try:
                average_precision = dict()
                for i in range(2):
                    s_precision[i], s_recall[i], _ = precision_recall_curve(y_test,
                                                                        predicted_array[:, i])
            except:
                tpr = dict()
                fpr = dict()
                roc = dict()
                # for i in range(2):
                #     fpr[i], tpr[i], _ = roc_curve(y_test,
                #                                   predicted_array[:, i])

        print('for depth ', depth, 'for split ', split, ' roc avg score - ', avg_roc_score / number_of_split,
              ' roc aupr score - ', avg_aupr_score / number_of_split)

n_precision = dict()
n_recall = dict()
for depth in np.arange(1, 10, 1):
    for split in np.arange(1, 10, 1):

        avg_roc_score = 0
        avg_aupr_score = 0
        # training process
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            X_train, y_train = sampler.fit_sample(X_train, y_train)

            X_train_reduced_feature_list = []
            X_test_reduced_feature_list = []

            predicted_list = []

            predicted_array = np.empty(shape=[len(y_test), 2])

            number_of_estimator = 0
            for feature in feature_division:

                number_of_estimator += 1

                # clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=split)
                clf = KNeighborsClassifier(n_neighbors=depth,leaf_size=split)

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
            try:
                avg_roc_score += roc_auc_score(y_test, predicted_array[:, 1])
                avg_aupr_score += average_precision_score(y_test, predicted_array[:, 1])
            except:
                print('')
            if avg_aupr_score > top_roc:
                top_roc = avg_aupr_score
            try:
                average_precision = dict()
                for i in range(2):
                    n_precision[i], n_recall[i], _ = precision_recall_curve(y_test,
                                                                        predicted_array[:, i])
            except:
                tpr = dict()
                fpr = dict()
                roc = dict()
                # for i in range(2):
                #     fpr[i], tpr[i], _ = roc_curve(y_test,
                #                                   predicted_array[:, i])

        print('for depth ', depth, 'for split ', split, ' roc avg score - ', avg_roc_score / number_of_split,
              ' roc aupr score - ', avg_aupr_score / number_of_split)




print('ploting', dataset)

plt.clf()
plt.plot(recall[1],precision[1], lw=2, color='red', label='Decision Tree')
plt.plot(e_recall[1],e_precision[1], lw=2, color='blue', label='Extra tree')
plt.plot(s_recall[1],s_precision[1], lw=2, color='black', label='SVM')
plt.plot(n_recall[1],n_precision[1], lw=2, color='green', label='K-nearest Neighbour')
# plt.plot(fpr[1], tpr[1], lw=2, color='navy', label='Roc')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Area under ROC curve')
plt.legend(loc="lower right")
plt.show()
# plt.savefig('/home/farshid/Desktop/roc/' + dataset + '.png')
