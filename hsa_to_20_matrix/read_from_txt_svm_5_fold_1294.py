import re
import random

import numpy
from sklearn import cross_validation,feature_selection
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


# from hsa_to_20_matrix import standardalize

def range_custom(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = int(291920)
# Total_data_number = 5000
# clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
data = []  # this list is to generate index value for k fold validation


file_read = open('/home/farshid/Desktop/dataset_test.txt', 'r')
count_for_number_of_instances = 0
i = 0
index = 0
# for i in range(0, 1287):
#     file_read.readline()
print("Starting To read From Text ...  ")
with open('dataset_test.txt', 'r') as file_read:
    for x in file_read:
        index += 1
        # x = file_read.readline()
        # print(x)
        if len(x) <= 10:
            break
        # l = re.findall("\d+\.\d+|-?0\.\d+|-?\d+", x)
        l = x.rstrip("\n").split(",")
        # print(l)
        l = list(map(float, l))
        # print(l[1286:1295])
        total_matrix.append(l)
        # b = l[1294]
        # if b == 1.0 :#and b != 0:
        #     i += 1
            # print( "count of 2 class " ,i, " value ",b)

        # if index % 10000 == 0:
        #     print(index)
        # break
        # if i == Total_data_number:
        #     break
print( "count of 2 class " ,i)
print("length of total matrix ", len(total_matrix) )
print("Starting To Standardize Total Matrix ...  ")
total_matrix = standardalize.std(total_matrix, 882, 412)

for l in total_matrix:
    feature_list_of_all_instances.append(l[0:1294])
    class_list_of_all_instances.append(l[1294])

for i in range(0, Total_data_number):
    data.append(i)

kf = cross_validation.KFold(Total_data_number, n_folds=5)

# Cs = numpy.logspace(-6, -1, 10)

# clf = GridSearchCV(estimator='svc',param_grid=dict(C = Cs) , n_jobs=-1 )


print("Starting K fold data to Svm   ...   ")
l = 0
for iteration, data in enumerate(kf, start=1):

    # print(iteration, data[0], data[1])
    train_set_indexes = data[0]
    test_set_indexes = data[1]

    temp_total_dataset = []

    temp_train_feature_list = []
    temp_train_class_list = []

    temp_test_feature_list = []
    temp_test_class_list = []

    counter_for_positive_class = 0

    print("Creating underSampled unbiased dataset")

    for index in train_set_indexes:
        if class_list_of_all_instances[index] == 1:
            temp_train_feature_list.append(feature_list_of_all_instances[index][0:1294])
            temp_train_class_list.append(class_list_of_all_instances[index])
            # temp_total_dataset.append(feature_list_of_all_instances[index][0:1282])
            counter_for_positive_class += 1

            # clf.fit(data[0],data[1])
    # print(counter_for_positive_class)


    randomly_generated_indexes = []

    while (counter_for_positive_class > 0):
        randomly_generated_indexes.append(int(random.uniform(0, len(data[0]))))
        counter_for_positive_class -= 1

    for cursor in randomly_generated_indexes:
        if class_list_of_all_instances[cursor] == 0:
            temp_train_feature_list.append(feature_list_of_all_instances[cursor][0:1294])
            temp_train_class_list.append(class_list_of_all_instances[cursor])
            # temp_total_dataset.append(feature_list_of_all_instances[index][0:1282])

    for index in test_set_indexes:
        temp_test_feature_list.append(feature_list_of_all_instances[index][0:1294])
        temp_test_class_list.append(class_list_of_all_instances[index])
        # temp_total_dataset.append(feature_list_of_all_instances[index][0:1282])

    # temp_total_dataset = sklearn.preprocessing.normalize(temp_total_dataset)

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

    C_List = []
    gamma_list = [2]
    for x in range(700,901,50):
        C_List.append(x)
    # weights = [0.1 , 0.9]
    # weights = np.asarray(weights)

    tuned_parameters = [{
        'kernel': ['rbf'],
        'gamma': gamma_list,
        # 'n_jobs':[-1],
        # 'cache_size': [100,110],
        'probability': [True],
        # 'tol': tol_range,
        'shrinking': [True],
        # 'class_weight': weights,
        'C': C_List,
    }, ]

    scores = ['average_precision', 'roc_auc', 'precision_macro', 'recall_macro', ]
    # scores = [ 'roc_auc','precision', 'precision_macro', 'recall_macro' ,'precision' ]

    transform = feature_selection.SelectPercentile(feature_selection.f_classif)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf1 = GridSearchCV(SVC(C=1) , tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5, scoring='%s' % score)

        clf = Pipeline([('anova', transform), ('svc', clf1)])
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

        print()
        print("Best parameters   " , clf1.best_params_)
        print("Best Score  ", clf1.best_score_)
        print("Best Estimator   ", clf1.best_estimator_)

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
        print("Average precision score :-", end='')
        print(average_precision_score(temp_test_class_list, predicted))

        # auc(temp_test_class_list,predicted)
        # plt.figure()
        # plt.plot(temp_test_class_list,predicted)
        # plt.show()
        # # if sklearn.metrics.roc_auc_score(temp_test_class_list, predicted) > max:
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
