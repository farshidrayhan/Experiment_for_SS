import re
import random

import gc
import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn import cross_validation,feature_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score, ShuffleSplit
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sympy.functions.special.gamma_functions import gamma
from sklearn.metrics import classification_report,average_precision_score
import standardalize


total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = 291920
# Total_data_number = 5000
data = []  # this list is to generate index value for k fold validation

print("Opening  Text ...  ")
count_for_number_of_instances = 0
i = 0
cout = 0
print("Starting To read From Text ...  ")
with open('/home/farshid/Desktop/dataset_1282.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
        l = list(map(float, l))
        # total_matrix.append(l)
        index = len(l) - 1
        # print(index)
        feature_list_of_all_instances.append(l[0:index])
        class_list_of_all_instances.append(l[index])
        # feature_list_of_all_instances.append(l[0:519])
        # class_list_of_all_instances.append(int(l[519]))
        i += 1
        #
        # if i == Total_data_number:
        #     break


print("Starting To Standardize Total Matrix ...  ")
feature_list_of_all_instances = standardalize.std(feature_list_of_all_instances, 882, 400)

c = 0

print("Total instances ", len(feature_list_of_all_instances))
print("Total Features  ", len(feature_list_of_all_instances[0]))

gc.collect()


# for l in total_matrix:
#     index = len(l) -1
#     # print(index)
#     feature_list_of_all_instances.append(l[0:index])
#     class_list_of_all_instances.append(l[index])

# total_matrix = []

gc.collect()

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)


kf = StratifiedKFold(n_splits=5,shuffle=True)
under_sample = RandomUnderSampler()

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

    gc.collect()
    #
    temp_train_feature_list, temp_train_class_list = under_sample.fit_sample(temp_train_feature_list,
                                                                             temp_train_class_list)
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

    # temp_train_feature_list = sklearn.preprocessing.normalize(temp_train_feature_list)
    # temp_train_feature_list =  StandardScaler().fit_transform( temp_train_feature_list )
    #
    # temp_test_feature_list = sklearn.preprocessing.normalize(temp_test_feature_list)
    # temp_test_feature_list =  StandardScaler().fit_transform(temp_test_feature_list)


    print("Training SVM ...  ")

    C_List = [3000]
    gamma_list = [.005]
    # for x in range(199,202):
    #     C_List.append(x)


    tuned_parameters = [{
        'kernel': ['rbf'],
        'gamma': gamma_list,
        # 'n_jobs':[-1],
        # 'cache_size': [100,110],
        'probability': [True],
        # 'tol': tol_range,
        'shrinking': [True],
        'C': C_List,
    }, ]

    scores = [ 'average_precision','roc_auc','precision_macro' ,'recall_macro',]
    scores = [ 'roc_auc' ]

    # transform = feature_selection.SelectPercentile(feature_selection.f_classif)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, n_jobs=4, pre_dispatch='3*n_jobs', cv=5, scoring='%s' % score)

        # clf = Pipeline([('anova',transform),('svc',clf1)])
        #                           ,n_jobs=-1
        # print("#############  ",clf.get_params())
        # # while tol <= 1:
        # clf = svm.SVC(C=100, cache_size=50, class_weight=None, coef0=0.0,
        #     decision_function_shape=None, degree=20, gamma=.0001, kernel='rbf',
        #     max_iter=-1, probability=True, random_state=None, shrinking=False,
        #     tol=0.25, verbose=False)  # use 20

        # clf = MLPClassifier(activation='logistic', learning_rate_init=c, hidden_layer_sizes=gamma, max_iter=3000,
        #                     warm_start=1
        #                     , tol=0.01, learning_rate='adaptive')

        # clf = svm.LinearSVC(C=.8,multi_class='crammer_singer')
        # tol += .05
        clf.fit(temp_train_feature_list, temp_train_class_list)
        # print("clf best score " + str(clf.best_score_) + "best estimator " + str(clf.best_estimator_.C) + "best pram " + str(clf.best_params_) )

        # print()
        # print("Best parameters   " , clf.best_params_)
        # print("Best Score  ", clf.best_score_)
        # print("Best Estimator   ", clf.best_estimator_)


        predicted = []
        j = 0
        for j in temp_test_feature_list:
            val = [j]
            predicted.append(clf.predict(val))
        # print("Classifier info  ", end='')
        # print(clf)
        # print("accuracy for predictions ", end='')
        # print(metrics.accuracy_score(temp_test_class_list, predicted))

        print("confusion matrix ")

        print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
        print("roc auc score :- ", end='')
        print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
        print("Average precision score :-",end='')
        print(average_precision_score(temp_test_class_list,predicted))
        # if sklearn.metrics.roc_auc_score(temp_test_class_list, predicted) > max:
        #     max = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)
        #     print("######################################################## updated auc ")
        # print("max auc ", end='')
        # print(max)
        # print("specificity ", end='')
        # specificity = float(confusion_matrix[0][0]) / (float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
        # print(specificity)
        # print("sensitivity ", end='')
        # sensitivity = float(confusion_matrix[1][1]) / float((confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
        # print(sensitivity)
        confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
        print("precision ", end='')
        precision = float(confusion_matrix[1][1]) / (float(confusion_matrix[1][1]) + float(confusion_matrix[0][1]))
        print(precision)
        print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))