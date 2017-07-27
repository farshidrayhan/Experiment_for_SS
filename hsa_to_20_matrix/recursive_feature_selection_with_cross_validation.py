"""
===================================================
Recursive feature elimination with cross-validation
===================================================

A recursive feature elimination example with automatic tuning of the
number of features selected with cross-validation.
"""
# print(__doc__)

import matplotlib.pyplot as plt
import numpy
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import standardalize

# Build a classification task using 3 informative features
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

with open('/home/farshid/Desktop/nr_dataset_1404.txt', 'r') as file_read:
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
        count_for_number_of_instances += 1
c = 0

print("Starting To Standardize Total Matrix ...  ")
#
total_matrix = standardalize.std(total_matrix, 882, 522)
print("Total instances ", len(total_matrix))
print("Total Features ", len(total_matrix[0])-1)

for l in total_matrix:
    index = len(l) -1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])

# X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                            n_redundant=2, n_repeated=0, n_classes=8,
#                            n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
for step in range(1,5):
    for c in numpy.arange(10, 5000, 50):
        for g in numpy.arange(.0001, .001, .0002):
            svc = SVC(kernel="linear",C=c,gamma=g)
            # The "accuracy" scoring is proportional to the number of correct
            # classifications

            print(" for c = " , c , " and g = ",g , " step = ", step)
            rfecv = RFECV(estimator=svc, step=step, cv=StratifiedKFold(5),
                          scoring='recall')
            rfecv.fit(feature_list_of_all_instances, class_list_of_all_instances)

            print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        # plt.figure()
        # plt.xlabel("Number of features selected")
        # plt.ylabel("Cross validation score (nb of correct classifications)")
        # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        # plt.show()
