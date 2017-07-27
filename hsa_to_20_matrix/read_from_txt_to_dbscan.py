import re
import random

from sklearn import svm, metrics
import sklearn
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import cross_val_predic
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np
from sklearn import datasets

from sklearn.preprocessing import StandardScaler

# print(__doc__)

feature_list = []
class_list = []
dict = []
index_list = []

i = 0
count_for_0_class = 0
total_1_class_instances = 2912
feature_list_of_all_instances = []
class_list_of_all_instances = []
# limit_on_instances = 2912 * 2
file_read = open('/home/farshid/Desktop/dataset.txt', 'r')
count_for_number_of_instances = 0
total_matrix = []
while i < 300000:
    x = file_read.readline()
    if len(x) <= 10:
        break
    l = re.findall("-?\d+", x)
    l = list(map(int, l))
    total_matrix.append(l)
    feature_list_of_all_instances.append(l[0:1282])
    class_list_of_all_instances.append(l[1282])

randomly_generated_indexes = []

while (count_for_0_class + 10 > 0):
    randomly_generated_indexes.append(int(random.uniform(0, len(total_matrix))))
    count_for_0_class -= 1

for cursor in randomly_generated_indexes:
    if total_matrix[cursor][1282] == 0:
        feature_list.append(total_matrix[cursor][0:1282])
        class_list.append(total_matrix[cursor][1282])

feature_list = np.asarray(feature_list)
class_list = np.asarray(class_list)
feature_list = StandardScaler().fit_transform(feature_list)
# feature_list_of_all_instances = np.asarray(feature_list_of_all_instances)
# total_matrix = np.asarray(total_matrix)
#

print("dbscan running .......")
eps = 1500
sample = 3000
while eps > 1:
    sample = 3000
    while sample > 1:
        clf = DBSCAN(eps=eps, min_samples=sample)
        #
        # #
        #
        #
        clf.fit(feature_list,class_list)

        # clf.fit(total_matrix)


        n_clusters_ = len(set(clf.labels_)) - (1 if -1 in clf.labels_ else 0)
        # print('Estimated number of clusters: %d' % n_clusters_)
        # print('eps size ' ,end = '')
        # print(eps)
        sample -= 50
        # if  n_clusters_ > 2 :

    print(clf.labels_)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('eps size ', end='')
    print(eps)

        # break


    eps += 1000  #
# # predicted = cross_val_predict(clf, iris.data, iris.target, cv=5)
# scores = cross_val_score(clf, feature_list, class_list, cv=5)
#
# # print ( metrics.accuracy_score(iris.target, predicted) )
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# if max_mean < scores.mean():
#     max_mean = scores.mean()

# predicted = cross_val_predict(clf, feature_list,class_list, cv=2)
# metrics.accuracy_score(class_list, predicted)

#
# predicted = []
# j = 0
# for j in range(0, len(feature_list)):
#     val = [feature_list[j]]
#     predicted.append(clf.fit_predict(val))
#
# # predicted.append( clf.predict([feature_list[1]]) )
#
# #   confusion matrix format
# #
# #   TN | FP
# #   FN | TP
# #
# #
# print("confusion matrix ")
# confusion_matrix = sklearn.metrics.confusion_matrix(class_list, predicted)
# print(confusion_matrix)
# print("specificity ", end='')
# specificity = float(confusion_matrix[0][0]) / (float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
# print(specificity)
# print("sensitivity ", end='')
# sensitivity = float(confusion_matrix[1][1]) / float((confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
# print(sensitivity)
# print("precision ", end='')
# precision = float(confusion_matrix[1][1]) / float(confusion_matrix[1][1]) + float(confusion_matrix[0][1])
# print(precision)
# # fpr tpr threshold
# print("roc curve :- fpr, tpr , threshold ")
# print(sklearn.metrics.roc_curve(class_list_of_all_instances, predicted))
#
# # print("\n\n\n\n max mean", max_mean)
