import re
import random
import plot_neighbourhood_cleaning_rule as ncr
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
# from sklearn import svm, metrics
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
print("Starting To read From Text ...  ")
with open('/home/farshid/Desktop/outF1_reduced.txt', 'r') as file_read:
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
# feature_list_of_all_instances = sklearn.preprocessing.normalize(feature_list_of_all_instances)

# feature_list_of_all_instances =  StandardScaler().fit_transform( feature_list_of_all_instances )

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)

print("Neighbourhood cleaning ...")
#
for neighbour in range (2,100,1):
    feature_list_of_all_instances, class_list_of_all_instances = ncr.neighbourhood_cleaning_rule(
        feature_list_of_all_instances, class_list_of_all_instances, neighbour)

    Total_data_number = len(feature_list_of_all_instances)
    print("New total instances  ",Total_data_number)

    randomized_list = []
    for x in range(0, Total_data_number):
        randomized_list.append(x)

    random.shuffle(randomized_list)

    temp_feature = []
    temp_class = []


    # for i in range(0,Total_data_number):
    for z in randomized_list:
        temp_feature.append( feature_list_of_all_instances[z] )
        temp_class.append( class_list_of_all_instances[z] )
        # continue
    feature_list_of_all_instances = temp_feature
    class_list_of_all_instances = temp_class

    # feature_list_of_all_instances = feature_list_of_all_instances.tolist()
    # class_list_of_all_instances = class_list_of_all_instances.tolist()
    data = []
    for i in range(0, Total_data_number):
        data.append(i)
    #
    kf = cross_validation.KFold(Total_data_number, n_folds=5,shuffle=True)
    #
    # # Cs = numpy.logspace(-6, -1, 10)
    #
    # # clf = GridSearchCV(estimator='svc',param_grid=dict(C = Cs) , n_jobs=-1 )
    #
    #
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

                temp_train_feature_list.append(feature_list_of_all_instances[index])
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
                temp_train_feature_list.append(feature_list_of_all_instances[cursor])
                temp_train_class_list.append(class_list_of_all_instances[cursor])
                # temp_total_dataset.append(feature_list_of_all_instances[index][0:1282])

        for index in test_set_indexes:
            temp_test_feature_list.append(feature_list_of_all_instances[index])
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

        C_List = [100]
        gamma_list = [.05, .10]
        # for x in range(99,101):
        #     C_List.append(x)

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

        # scores = ['average_precision', 'roc_auc', 'precision_macro', 'recall_macro' ]
        # scores = [ 'roc_auc','precision', 'precision_macro', 'recall_macro' ,'precision' ]
        scores = ['average_precision' ]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1) , tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5, scoring='%s' % score)
            #                           ,n_jobs=-1
            # print("#############  ",clf.get_params())
            # # while tol <= 1:
            # clf = svm.SVC(C=0.392, cache_size=50, class_weight=None, coef0=0.0,
            #     decision_function_shape=None, degree=20, gamma=gamma, kernel='rbf',
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
            print("Best parameters   ", clf.best_params_)
            print("Best Score  ", clf.best_score_)
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


            break
        break