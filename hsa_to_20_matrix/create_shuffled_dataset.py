import re
import random
import numpy as np
from sklearn import cross_validation

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
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
        i += 1
        #
        # if i == Total_data_number:
        #     break

c = 0

print("Total instances ", len(total_matrix))

for l in total_matrix:
    index = len(l) -1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])
print("Total features ",len(feature_list_of_all_instances[0]))

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)

number_of_folds = 5
kf = StratifiedKFold(n_splits=number_of_folds,shuffle=True)


file = open('/home/farshid/Desktop/nr_dataset_shuffled_1294.txt', 'w')


for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):

    # temp_train_feature_list = feature_list_of_all_instances[train_set_indexes]
    # temp_train_class_list = class_list_of_all_instances[train_set_indexes]
    temp_train_feature_list = []
    temp_train_class_list = []
    for index in train_set_indexes:
        feature_list_of_all_instances.append(feature_list_of_all_instances[index])
        class_list_of_all_instances.append(class_list_of_all_instances[index])

    temp_test_feature_list = []
    temp_test_class_list = []
    for index in test_set_indexes:
        feature_list_of_all_instances.append(feature_list_of_all_instances[index])
        class_list_of_all_instances.append(class_list_of_all_instances[index])

    break

print(len(feature_list_of_all_instances[0]))

for j in range(0,len(feature_list_of_all_instances)):

    feature_list_of_all_instances[j].append(class_list_of_all_instances[j])

print(len(feature_list_of_all_instances[0]))
print()
