# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
import numpy
from sklearn.model_selection import StratifiedKFold

print(__doc__)

import numpy as np

import random
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import standardalize

##############################################################################
# Generate sample data
centers = [[.5, 0], [0, 1], [-.5, 0], [-1, -1], [-1.5, 0]]
total_points = 1000
X, labels_true = make_blobs(n_samples=total_points, centers=centers, cluster_std=0.2,
                            random_state=0)
Y = np.array([[random.uniform(0, 5), random.uniform(0, 5)]] * 1000)
X = StandardScaler().fit_transform(X)
##############################################################################
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
print("Starting To read From Text ...  ")
with open('/home/farshid/Desktop/nr_dataset_1294.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
        l = list(map(float, l))
        total_matrix.append(l)
        # feature_list_of_all_instances.append(l[0:519])
        # class_list_of_all_instances.append(int(l[519]))
        # i += 1
        #
        # if i == Total_data_number:
        #     break

c = 0

# print("Starting To Standardize Total Matrix ...  ")

# total_matrix = standardalize.std(total_matrix, 882, 400)
print("Total instances ", len(total_matrix))
total_points = len(total_matrix)
for l in total_matrix:
    index = len(l) - 1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])

kf = StratifiedKFold(n_splits=5, shuffle=True)
print("Starting K fold data to Classifier   ...   ")
temp_test_class_list = []
for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):

    # temp_train_feature_list = feature_list_of_all_instances[train_set_indexes]
    # temp_train_class_list = class_list_of_all_instances[train_set_indexes]
    temp_train_feature_list = []
    temp_train_class_list = []
    for index in train_set_indexes:
        temp_train_feature_list.append(feature_list_of_all_instances[index])
        temp_train_class_list.append(class_list_of_all_instances[index])

    temp_test_feature_list = []
    temp_test_class_list = []
    for index in test_set_indexes:
        temp_test_feature_list.append(feature_list_of_all_instances[index])
        temp_test_class_list.append(class_list_of_all_instances[index])

    under_sample = RandomUnderSampler()
    temp_train_feature_list, temp_train_class_list = under_sample.fit_sample(temp_train_feature_list,
                                                                             temp_train_class_list)

    X = StandardScaler().fit_transform(temp_train_feature_list)
    ##############################################################################
    # Compute DBSCAN
    best_eps = 0
    best_min = 0
    prev_clustered_area = 0
    prev_clusters = 99999999999


    for eps in numpy.arange(7.87,100,.01): # eps=35 min =2  4 cluster 40 2 2 cluster
        for min in numpy.arange(3,5,1):

            print("For eps  ", eps, " and min ", min)

            db = DBSCAN(eps=eps, min_samples=min,n_jobs=4).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            # print(db.labels_)
            error = 0
            for i in db.labels_:
                if i == -1:
                    error += 1

            # print("Total points ", total_points)
            # print("Total outliers ", error)
            # print("Clustered  Area ", (1 - error / total_points) * 100, "%")
            # print('Estimated number of clusters: %d' % n_clusters_)

            temp_class_one_list = []
            for i in range(0,len(temp_test_feature_list)):
                if temp_test_class_list[i] == 1:
                    temp_class_one_list.append(temp_test_feature_list[i])

            predicted = db.fit_predict(temp_class_one_list)

            if len(set(predicted)) > 2:

                print("For eps  ", eps, " and min ", min)


                print("Total points ", total_points)
                print("Total outliers ", error)
                print("Clustered  Area ", (1 - error / total_points) * 100, "%")
                print('Estimated number of clusters: %d' % n_clusters_)

                print("lenght ",len(predicted) , "data ",predicted)
                print("lenght ",len(set(predicted)) , "data ",set(predicted))
                # print()

                outliers = 0
                for v in predicted:
                    if v == -1:
                        outliers += 1

                print("Total unique clusters for class 1 data in train set ", len(predicted)-outliers)
            # print("Total unique clusters for class 1 data in train set ", len(set(predicted)))

            # if  prev_clusters < len(set(predicted)) and len(set(predicted)) > 1 :
            #     best_eps = eps
            #     best_min = min
            # print("best eps  ", best_eps, " and min ", best_min)
            # print()

    break
#  print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

##############################################################################
# # Plot result
# import matplotlib.pyplot as plt
#
# # Black removed and is used for noise instead.
#
#
# # array =  (np.array([ 0.        ,  0.26729364,  1.        ]), np.array([ 0.        ,  0.72321429,  1.        ]), np.array([2, 1, 0]))
# # plt.plot(array)
# # plt.show()
#
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=6)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
