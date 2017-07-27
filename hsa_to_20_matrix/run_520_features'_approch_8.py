import re
import random
import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn import cross_validation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from   sklearn.ensemble import RandomForestClassifier
# from brew.base import Ensemble
from sklearn.model_selection import StratifiedKFold
import sklearn
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

print("Opening  Text ...  ")
count_for_number_of_instances = 0
i = 0
cout = 0
print("Opening  Text ...  ")

# file_read = open('feature_reduced_dataset.txt.txt', 'r')
count_for_number_of_instances = 0
i = 0
cout = 0
# for i in range(0, 1287):
#     file_read.readline()
z = 1
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


data = []
for i in range(0, Total_data_number):
    data.append(i)

number_of_folds = 5
kf = StratifiedKFold(n_splits=number_of_folds,shuffle=True)
# sampler = RandomUnderSampler()

print("Starting K fold data to Classifier   ...   ")

top_roc  = 0
top_avg_roc = 0
avg_roc = 0
for simulation in range(0,1):
    avg_roc = 0
    print("Simulation no  ",simulation)
    print()

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

        # sampler = ADASYN()
        # sampler  = SMOTE(kind='svm')
        #
        # temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
        #                                                                          temp_train_class_list)

        sampler = RandomUnderSampler()
        temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
                                                                            temp_train_class_list)

        # temp_train_class_list = temp_train_class_list.tolist()
        # temp_train_feature_list = temp_train_feature_list.tolist()
        #
        #
        # randomly_generated_indexes = []
        # size = len(temp_train_feature_list)
        # while (size > 0):
        #     randomly_generated_indexes.append(int(random.uniform(0, len(feature_list_of_all_instances[0]))))
        #     size -= 1
        #
        # for cursor in randomly_generated_indexes:
        #     if class_list_of_all_instances[cursor] == 0:
        #         temp_train_feature_list.append(feature_list_of_all_instances[cursor])
        #         temp_train_class_list.append(class_list_of_all_instances[cursor])

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


        tuned_parameters = [{
            # 'n_init': [5,10,20] ,
            # 'n_clusters': [2,10,20,50],
            # 'init': ['k-means++','random'],
            # 'algorithm': ['auto'],
            # 'max_iter' : [300,400,500],
            # 'precompute_distances':[True,False]
        }, ]
        #
        scores = [ 'roc_auc' , 'recall_macro','precision_macro' , 'average_precision']
        scores = [ 'roc_auc' ]

        #
        #
        for score in scores:
            print("Tuning hyper-parameters for %s" % score)

            clf = GridSearchCV(AgglomerativeClustering(), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5)
                               # scoring='%s' % score)
            # clf = AdaBoostClassifier(estimator, n_estimators=50, learning_rate=1, algorithm='SAMME')

            clf.fit(temp_train_feature_list)

            print("Best Score  ", clf.best_score_)
            print("Best Parameter   ", clf.best_params_)
            # print("Best Estimator   ", clf.best_estimator_)

            print("Predicting ")
            predicted = clf.fit_predict(temp_test_feature_list)

            print(" Confusion matrix ")
            print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
            print("# Roc auc score           :- ", end='')
            print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
            print("# Average precision score :- ", end='')
            print(average_precision_score(temp_test_class_list, predicted))
            fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
            print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
            # confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
            # print(" -> Specificity ", end='')
            # specificity = float(confusion_matrix[0][0]) / (float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
            # print(specificity)
            # print(" -> Sensitivity ", end='')
            # sensitivity = float(confusion_matrix[1][1]) / float((confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
            # print(sensitivity)

            print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))

            avg_roc += sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)


            if top_roc < sklearn.metrics.roc_auc_score(temp_test_class_list, predicted) :
                top_roc = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)


            # print("####  Average roc :- ", avg_roc , end='\n\n')

        print("Top roc     :- " , top_roc , end='\n\n')
    print("#### top Average roc :- ", top_avg_roc, end='\n\n')
    if top_avg_roc < avg_roc / number_of_folds :
        top_avg_roc = avg_roc / number_of_folds
        print("#### top Average roc :- ", top_avg_roc ,end='\n\n')