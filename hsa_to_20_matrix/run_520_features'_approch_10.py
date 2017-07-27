import re
import random
import numpy as np
import sklearn
from builtins import range
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling.random_over_sampler import RandomOverSampler
from imblearn.under_sampling import TomekLinks
from nltk.tbl import feature
from pandas.parser import k
from sklearn import cross_validation
import logging

import plot_neighbourhood_cleaning_rule as ncr
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, InstanceHardnessThreshold, AllKNN, \
    neighbourhood_cleaning_rule
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from   sklearn.ensemble import RandomForestClassifier
# from brew.base import Ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, GridSearchCV
# from sklearn.model_selection import cross_val_predict, cross_val_score, ShuffleSplit
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier
from sympy.functions.special.gamma_functions import gamma
import standardalize

total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = 291920
# Total_data_number = 5000
data = []  # this list is to generate index value for k fold validation

# logging.basicConfig(filename='/home/farshid/Desktop/run_520_features_approch_2.log' ,level=print,
#                     format='%(asctime)s:%(levelname)s:%(message)s')
features = []
with open('/home/farshid/Desktop/cols.txt', 'r') as file_read:
    for x in file_read:
        # if len(x) <= 10:
        #     break
        l = x.rstrip('\n').split(',')

        l = list(map(int, l))
        # l = ''.join(l)
        features.append(l)


print("Opening  Text ...  ")
count_for_number_of_instances = 0
i = 0
cout = 0
print("Starting To read From Text ...  ")
with open('/home/farshid/Desktop/outF2BG_dataset_feature_reduced.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
        l = list(map(float, l))
        total_matrix.append(l)
        # feature_list_of_all_instances.append(l[0:519])
        # class_list_of_all_instances.append(int(l[519]))
        i += 1
        #
        # if i == Total_data_number:
        #     break

c = 0
# print("Starting To Standardize Total Matrix ...  ")

# total_matrix = standardalize.std(total_matrix, 882, 522+73)
print("Total instances ", len(total_matrix))
print("Total Features ", len(total_matrix[0]) - 1)

for l in total_matrix:
    index = len(l) - 1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])

class_list_of_all_instances = [0 if x == -1 else x for x in class_list_of_all_instances]

# temp_features = [[0 for x in range(len(features) ) ] for y in range(len(total_matrix))]
temp_features = [[0 for x in range(0,1282 ) ] for y in range(len(feature_list_of_all_instances))]

# feature_list_of_all_instances =  StandardScaler().fit_transform( feature_list_of_all_instances )

for i in range(0,len(feature_list_of_all_instances)):
    index = 0
    for j in range(0 ,len(feature_list_of_all_instances[i] ) ) :
        if j < 1282:
            temp_features[i][j] = feature_list_of_all_instances[i][j]

# feature_list_of_all_instances = temp_features

#
feature_list_of_all_instances = sklearn.preprocessing.normalize(feature_list_of_all_instances)

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)



for i in range(0,len(feature_list_of_all_instances)):
    index = 0
    for j in range(0 ,len(feature_list_of_all_instances[i] ) ) :
        if [j] in features:
            # continue
            # print("index " , index , " ")
            temp_features[i][index] = feature_list_of_all_instances[i][j]
            index += 1


# feature_list_of_all_instances = temp_features

number_of_folds = 5
kf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)

# sampler = RandomUnderSampler(ratio=.99)
sampler =  SMOTE()
# sampler = ClusterCentroids(estimator=KMeans(n_clusters= 4) )
# sampler = InstanceHardnessThreshold(estimator=RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=5))
# sampler = AllKNN(n_neighbors=50)
# sampler = SMOTEENN(k = 4,m=20,n_neighbors=3,enn=5)
# sampler =  ADASYN(k=3,n_neighbors=3)
# sampler =  RandomOverSampler(ratio=.6)

#

print("Starting K fold data to Classifier   ...   ")
avg_roc = 0
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

    counter_for_positive_class = 0

    print("Creating Training dataset")

    # sampler = RandomOverSampler(ratio=.6)

    # temp_train_feature_list, temp_train_class_list = ncr.neighbourhood_cleaning_rule(
    #     temp_train_feature_list, temp_train_class_list, 2)

    temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
                                                                        temp_train_class_list)
    # temp_train_feature_list, temp_train_class_list = ncr.neighbourhood_cleaning_rule(
    #     temp_train_feature_list, temp_train_class_list, 2)

    # sampler = RandomUnderSampler()
    # temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
    #                                                                     temp_train_class_list)
    # temp_train_feature_list, temp_train_class_list = ncr.neighbourhood_cleaning_rule(
    #     temp_train_feature_list, temp_train_class_list, 2)
    #
    # sampler = ADASYN()
    # temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
    #                                                                     temp_train_class_list)

    #
    # temp_train_feature_list, temp_train_class_list = TomekLinks().fit_sample(temp_train_feature_list,
    #                                                                          temp_train_class_list)
    # temp_train_feature_list, temp_train_class_list = NearMiss(n_neighbors=2).fit_sample(temp_train_feature_list, temp_train_class_list)
    #


    cou1 = 0
    cou2 = 0
    for h in temp_train_class_list:
        if h == 1:
            cou1 += 1
        if h == 0:
            cou2 += 1
    print("positive in train list ", cou1)
    print("negative in train list ", cou2)

    cou1 = 0
    cou2 = 0
    for h in temp_test_class_list:
        if h == 1:
            cou1 += 1
        if h == 0:
            cou2 += 1
    print("positive in test list ", cou1)
    print("negative in test list ", cou2)

    print("Training ...  ")

    #
    # est = DecisionTreeClassifier(max_depth=15, min_samples_split=6)   # roc avg result  = 0.895951148384 cross fold = 5 l_rate = 1 n_estimator = 150
    # est1 =  DecisionTreeClassifier(max_depth=14, min_samples_split=8)
    #
    # estimator_list = []
    # for d in range(4,5,1):
    #     for s in range(2,5,1):
    #         estimator_list.append(DecisionTreeClassifier(max_depth=d,min_samples_split=s))
    # learning_rate = [1]
    #
    # tuned_parameters = [{
    #     'base_estimator': estimator_list,
    #     'n_estimators': [100],
    #     'learning_rate': [1],  # 'entropy'],
    #     'algorithm': ['SAMME'],
    # }, ]

    # scores = [ 'roc_auc' , 'recall_macro','precision_macro' , 'average_precision']
    # scores = [ 'roc_auc' ]
    #


    # for score in scores:
    #     print("Tuning hyper-parameters for %s" % score)
    #
    #     clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=10,
    #                        scoring='%s' % score)
    clf = GaussianNB()

    clf.fit(temp_train_feature_list, temp_train_class_list)

    # print("Best Score  ", clf.best_score_)
    # print("Best Estimator   ", clf.best_estimator_)
    #
    print("Predicting ")
    predicted = clf.predict_proba(temp_test_feature_list)

    # predicted_with_threshold = []
    # j = 0
    # for j in temp_test_feature_list:
    #     val = [j]
    #     val1 = clf.predict_proba(val)
    #     # print(val1)
    #     if val1[0][1] > .80:
    #         predicted_with_threshold.append(1)
    #     else:
    #         predicted_with_threshold.append(0)

    # print(" Confusion matrix ")
    # print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
    # print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted_with_threshold))


    print("# Roc auc score                :- ", end='')
    print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:, 1]))
    # print("# Average precision score      :- ", end='')
    # print(average_precision_score(temp_test_class_list, predicted))
    # print("# Roc auc score with threshold :- ", end='')
    # print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted_with_threshold))
    #
    avg_roc += sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:,1])
    # fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
    # print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
    # confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
    # print(" -> Specificity ", end='')
    # specificity = float(confusion_matrix[0][0]) / (float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
    # print(specificity)
    # print(" -> Sensitivity ", end='')
    # sensitivity = float(confusion_matrix[1][1]) / float((confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
    # print(sensitivity)
    # print("Actual Report ")
    # print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))
    # print("Report with threshold ")
    # print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted_with_threshold)))
    # avg_roc += sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)
    # break

print("Average roc = ", avg_roc / number_of_folds)
