import re
import random
import plot_neighbourhood_cleaning_rule as ncr
import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn import cross_validation
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from   sklearn.ensemble import RandomForestClassifier

# from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.model_selection import cross_val_predict, GridSearchCV
# from sklearn.model_selection import cross_val_predict, cross_val_score, ShuffleSplit
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler, normalize
from sympy.functions.special.gamma_functions import gamma
import standardalize

# from hsa_to_20_matrix import standardalize

total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = 291920
# Total_data_number = 5000
# clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
data = []  # this list is to generate index value for k fold validation
print("Opening  Text ...  ")

# file_read = open('feature_reduced_dataset.txt.txt', 'r')
count_for_number_of_instances = 0
i = 0
cout = 0
# for i in range(0, 1287):
#     file_read.readline()
z = 1
print("Starting To read From Text ...  ")
with open('/home/farshid/Desktop/ic_dataset_1282.txt', 'r') as file_read:
    for x in file_read:
        # z += 1
        # if z <= 1292:
        #     continue
        if len(x) <= 10:
            break
        # print(x)
        l = x.rstrip('\n').split(',')
        # print(len(l))
        # l = re.findall("\d+\.\d+|-?0\.\d+|-?\d+", l)
        # print(l)
        l = list(map(float, l))
        # l = list(map(int, l))
        # print(l[519])
        # if l[519] == 1:
        #     cout += 1
        #     print(cout)
        total_matrix.append(l)
        i += 1
        #
        if i == Total_data_number:
            break
        # break


c = 0

print("Total instances ", len(total_matrix))

for l in total_matrix:
    last_index = len(l) - 1
    feature_list_of_all_instances.append(l[0:last_index])
    class_list_of_all_instances.append(l[last_index])


class_list_of_all_instances =  [0 if x==-1 else x for x in class_list_of_all_instances]
# c=0
print("Total features ", len(feature_list_of_all_instances[0]))
feature_list_of_all_instances = sklearn.preprocessing.normalize(feature_list_of_all_instances)

# feature_list_of_all_instances =  StandardScaler().fit_transform( feature_list_of_all_instances )

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)

######################################################################################################################
# Here the approach is  to use grid search for random forest on dataset until the goal scores are reached #
######################################################################################################################

# feature_list_of_all_instances, class_list_of_all_instances = ncr.neighbourhood_cleaning_rule(
#     feature_list_of_all_instances, class_list_of_all_instances, neighbour)
#
# Total_data_number = len(feature_list_of_all_instances)
# print("for ", iter, "th iterationNew total instances  ", Total_data_number)
#
# randomized_list = []
# for x in range(0, Total_data_number):
#     randomized_list.append(x)
#
# random.shuffle(randomized_list)

# temp_feature = []
# temp_class = []

# for i in range(0,Total_data_number):
# for z in randomized_list:
#     temp_feature.append(feature_list_of_all_instances[z])
#     temp_class.append(class_list_of_all_instances[z])
#     # continue
# feature_list_of_all_instances = temp_feature
# class_list_of_all_instances = temp_class
#
# iter += 1
# if iter <= 70:
#     continue

# feature_list_of_all_instances = feature_list_of_all_instances.tolist()
# class_list_of_all_instances = class_list_of_all_instances.tolist()
data = []
for i in range(0, Total_data_number):
    data.append(i)
#
kf = StratifiedKFold(n_splits=5,shuffle=True)

sampler = RandomUnderSampler(ratio=.99)
# sampler =  ADASYN(k=2,n_neighbors=2)
# sampler = RandomOverSampler(ratio=.7)
# sampler =  SMOTE(ratio=.8)
# sampler = ClusterCentroids(estimator=KMeans(n_clusters= 10) )
# sampler = InstanceHardnessThreshold(estimator=RandomForestClassifier(n_estimators=150,max_depth=18,min_samples_split=8))
# sampler = AllKNN(n_neighbors=10)
# sampler = SMOTEENN(k = 3,m=3,n_neighbors=3,enn=2)

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



    # temp_test_feature_list = feature_list_of_all_instances[temp_test_feature_list]
    # temp_test_class_list = class_list_of_all_instances[temp_test_feature_list]

    counter_for_positive_class = 0

    print("Creating Training dataset")

    temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
                                                                             temp_train_class_list)

    temp_train_feature_list, temp_train_class_list = ncr.neighbourhood_cleaning_rule(
        temp_train_feature_list, temp_train_class_list, 2)
    #
    # temp_train_feature_list, temp_train_class_list = TomekLinks().fit_sample(temp_train_feature_list, temp_train_class_list)
    # temp_train_feature_list, temp_train_class_list = NearMiss(n_neighbors=3).fit_sample(temp_train_feature_list, temp_train_class_list)


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


    print("Training...  ")


    # tuned_parameters = [{
    #     'kernel': ['rbf'],
    #     'gamma': gamma_list,
    #
    #     # 'n_jobs':[-1],
    #     # 'cache_size': [100,110],
    #     'probability': [True],
    #     # 'decision_function_shape': ['None', 'ovo', 'ovr'],
    #     # 'tol': tol_range,
    #     'shrinking': [True],
    #     # 'max_iter' : iteration_list,
    #     'class_weight': ['balanced',None],
    #     'C': C_List,
    # }, ]
    tuned_parameters = [{
        'n_estimators': [ 7],
        'criterion': ['gini' , 'entropy'],  # 'entropy'],
        'max_features': [20,30],
        'max_depth': [15,18],
        'min_samples_split': [5, 6],
        'min_samples_leaf': [4],
        'bootstrap': [True],
        'n_jobs': [4],
        'class_weight': [None],
        # 'C': C_List,
    }, ]
    # tuned_parameters = [{
    #     'activation' : ['logistic','relu'],
    #     'learning_rate_init': [.3,.5,.8],
    #     'alpha' : [0.0001,.001],
    #     'warm_start':[True],
    #     'learning_rate': ['adaptive','invscaling'],
    #     'hidden_layer_sizes': [100,300,400],
    #     'tol':[.001],
    #     # 'decision_function_shape': ['None', 'ovo', 'ovr'],
    #     # 'tol': tol_range,
    #     # 'shrinking': [True],
    #     'max_iter': [200,100],
    #     # 'class_weight': weights,
    #     # 'C': C_List,
    # }, ]

    scores = ['roc_auc', 'average_precision', 'precision_macro', 'recall_macro']
    # scores = [ 'roc_auc','precision', 'precision_macro', '' ,'precision' ]
    scores = ['recall_macro']
    # for it in range(1,100,2):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        #     print()
        #     print("training for ",it , " iterations ")
        # randomized_list = []
        # for x in range(0, Total_data_number):
        #     randomized_list.append(x)
        #
        # random.shuffle(randomized_list)

        # print("Neighbourhood cleaning ...")
        # neighbour = 5
        # iter = 0
        # while len(temp_train_feature_list) >= 2912*2 :
        #     temp_train_feature_list, temp_train_class_list = ncr.neighbourhood_cleaning_rule(
        #         temp_train_feature_list, temp_train_class_list, neighbour)
        #
        #
        #     print("for ", iter, "th iterationNew total instances  ", Total_data_number)


        #
        # clf = GridSearchCV(SVC(C=1), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5,
        #                    scoring='%s' % score)
        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5,
                           scoring='%s' % score)
        #                           ,n_jobs=-1

        # clf = GridSearchCV(MLPClassifier(), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5,
        #                    scoring='%s' % score)


        # print("#############  ",clf.get_params())
        # # while tol <= 1:
        # clf = sklearn.svm.SVC(C=8000,  class_weight=None, coef0=0.0,
        #                       decision_function_shape=None,  gamma=20, kernel='rbf',
        #                       max_iter=it, probability=True, random_state=None, shrinking=True,
        #                        verbose=False)  # use 20

        # clf = MLPClassifier(activation='logistic', learning_rate_init=c, hidden_layer_sizes=20, max_iter=it,
        #                     warm_start=1
        #                     , tol=0.01, learning_rate='adaptive')

        # clf = svm.LinearSVC(C=.8,multi_class='crammer_singer')
        # tol += .05
        clf.fit(temp_train_feature_list, temp_train_class_list)
        # print("clf best score " + str(clf.best_score_) + "best estimator " + str(clf.best_estimator_.C) + "best pram " + str(clf.best_params_) )

        # print()
        # print("Best parameters   ", clf.best_params_)
        print("Best Score  ", clf.best_score_)
        print("Best Estimator   ", clf.best_estimator_)

        print("Predicting ")
        predicted = clf.predict(temp_test_feature_list)

        # print("Classifier info  ", end='')
        # print(clf)
        # print("accuracy for predictions ", end='')
        # print(metrics.accuracy_score(temp_test_class_list, predicted))
        # print(" Prediction without probability")
        print(" Confusion matrix ")
        print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
        print("# Roc auc score           :- ", end='')
        print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
        print("# Average precision score :- ", end='')
        print(average_precision_score(temp_test_class_list, predicted))
        fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
        print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
        confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
        # print(" -> Specificity ", end='')
        # specificity = float(confusion_matrix[0][0]) / (
        #     float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
        # print(specificity)
        # print(" -> Sensitivity ", end='')
        # sensitivity = float(confusion_matrix[1][1]) / float(
        #     (confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
        # print(sensitivity)
        # print(" -> Precision ", end='')
        # precision = float(confusion_matrix[1][1]) / (float(confusion_matrix[1][1]) + float(confusion_matrix[0][1]))
        # print(precision)
        print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))

        # if confusion_matrix[1][1] == 0:
        #     break

        # predicted = []
        # j = 0
        # for j in temp_test_feature_list:
        #     val = [j]
        #
        #     predicted.append(clf.predict_proba(val))
        # auc(temp_test_class_list,predicted)
        # plt.figure()
        # plt.plot(temp_test_class_list,predicted)
        # plt.show()
        # # if sklearn.metrics.roc_auc_score(temp_test_class_list, predicted) > max:
        #     max = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)
        #     print("######################################################## updated auc ")
        # print("max auc ", end='')
        # print(max)
        # print(" Prediction with probability")
        # print(" Confusion matrix ")
        # print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
        # print("# Roc auc score           :- ", end='')
        # print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
        # print("# Average precision score :- ", end='')
        # print(average_precision_score(temp_test_class_list, predicted))
        # fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
        # print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
        # confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
        # print(" -> Specificity ", end='')
        # specificity = float(confusion_matrix[0][0]) / (
        #     float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
        # print(specificity)
        # print(" -> Sensitivity ", end='')
        # sensitivity = float(confusion_matrix[1][1]) / float(
        #     (confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
        # print(sensitivity)
        # print(" -> Precision ", end='')
        # precision = float(confusion_matrix[1][1]) / (
        #     float(confusion_matrix[1][1]) + float(confusion_matrix[0][1]))
        # print(precision)
        # print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))


        # break
        # break
