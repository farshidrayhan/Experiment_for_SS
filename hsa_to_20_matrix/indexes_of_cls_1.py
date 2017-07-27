import re
import random
import plot_neighbourhood_cleaning_rule as ncr
# import numpy
from sklearn import cross_validation
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
with open('feature_reduced_dataset.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
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
