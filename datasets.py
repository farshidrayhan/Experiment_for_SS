from numpy import transpose
from sklearn import cross_validation, svm
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
import sklearn.svm
from sklearn.cross_validation import KFold

from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.svm import SVC


def range_custom(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


# digits = make_classification(n_samples=10,n_features=6,n_informative=3,n_redundant=2,n_repeated=1,n_classes=3,n_clusters_per_class=2)
# print(digits[0])
# print(digits[1])
# feature_matrix = digits[0]
# class_matrix = digits[1]


feature_matrix = [
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1],

]
class_list = [1, 0, 1, 1, 0]
red_list = []
for i in range(0, 5):
    if (all(value[i] == 0 for value in feature_matrix)):
        print(i, "th feature redundant")
        red_list.append(i)
        continue
    if (all(value[i] == 1 for value in feature_matrix)):
        red_list.append(i)
        print(i, "th feature redundant")

count = 0
for i in range(0, 6):
    count = 0
    for j in range(0, 5):
        if class_list[j] == feature_matrix[j][i]:
            count += 1

    print(i, "th matched ", count / 5 * 100, "% with class data")

# row = []
# for row in feature_matrix:
#     print(row[0])

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
b = [1, 2, 3, 4, 7, 8, 9]
c = []
for i in a:

    if i not in b:
        c.append(i)

# print(c)
#
for i in range(0, 5):
    for j in range(0, 6):
        print(feature_matrix[i][j], end='')

    print()

feature_matrix = transpose(feature_matrix)
feature_matrix = feature_matrix.tolist()
print()
reduced_feature_list = []
for i in range(0, 6):
    if i not in red_list:
        reduced_feature_list.append(feature_matrix[i])

feature_matrix = transpose(reduced_feature_list)
feature_matrix = feature_matrix.tolist()

# for i in range(0, 2):
#     for j in range(0, 2):
#         print(feature_matrix[i][j], end='')
#
#     print()

print(feature_matrix)



# #
# for i in range(0, 5):
#     for j in range(0, 6):
#         if red_list.__contains__(j):
#             feature_matrix[i].__delitem__(j)

# print(feature_matrix)
#
# indexes = [0,2,3 ]
# print(a)
# for i in a:
#     for index in sorted(indexes, reverse=True):
#         del a[index]
#
#
#
# print(a)

# for i in range(0, 3):
#     for j in range(0, 6):
#         print(feature_matrix[i][j], end='')

# print(feature_matrix)


















# kf = cross_validation.KFold(150, n_folds=5)
#
#
# for iteration, data in enumerate(kf, start=0):
#
#     # print(iteration, data[0], data[1])
#     train_set_indexes = data[0]
#     test_set_indexes = data[1]
#
#     # print(train_set_indexes)
#     # print(test_set_indexes)
#
#     train_feature_list = []
#     train_class_list = []
#
#     test_feature_list = []
#     test_class_list = []
#
#
#     for index in train_set_indexes:
#         train_feature_list.append(feature_matrix[index])
#         train_class_list.append(class_matrix[index])
#
#     for index in test_set_indexes:
#         test_feature_list.append(feature_matrix[index])
#         test_class_list.append(class_matrix[index])
#
#     cou1 = 0
#     cou2 = 0
#     for h in train_class_list:
#         if h == 1:
#             cou1 += 1
#         if h == 0:
#             cou2 += 1
#     print("positive in train list ", cou1)
#     print("negative in train list ", cou2)
#
#     cou1 = 0
#     cou2 = 0
#     for h in test_class_list:
#         if h == 1:
#             cou1 += 1
#         if h == 0:
#             cou2 += 1
#     print("positive in test list ", cou1)
#     print("negative in test list ", cou2)
#
#     C_List = []
#     gamma_list = []
#     tol_range = []
#
#     for x in range(1, 10):
#         C_List.append(x)
#     for x in range_custom(.0001, .001, .0001):
#         gamma_list.append(x)
#     for x in range_custom(.01, 3, .05):
#         tol_range.append(x)
#
#     tuned_parameters = [{
#         'kernel': ['rbf'],
#         'gamma': gamma_list,
#         # 'n_jobs':[-1],
#         # 'cache_size': [100,110],
#         'probability': [True,False],
#         'tol': tol_range,
#         'shrinking': [True,False],
#         'C': C_List,
#     }, ]
#
#     scores = ['average_precision', 'roc_auc', 'precision_macro', 'recall_macro', 'f1_macro']
#     # scores = [ 'roc_auc','precision', 'precision_macro', 'recall_macro' ,'precision' ]
#
#     for score in scores:
#         print("# Tuning hyper-parameters for %s" % score)
#         print()
#         clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s' % score)
#
#         clf.fit(train_feature_list,train_class_list)
#
#         print("Best parameters   ", clf.best_params_)
#
#         predicted = []
#         j = 0
#         for j in test_feature_list:
#             val = [j]
#             predicted.append(clf.predict(val))
#
#         print("confusion matrix ")
#
#         print(sklearn.metrics.confusion_matrix(test_class_list, predicted))
#         print("roc auc score :- ", end='')
#         print(sklearn.metrics.roc_auc_score(test_class_list, predicted))
