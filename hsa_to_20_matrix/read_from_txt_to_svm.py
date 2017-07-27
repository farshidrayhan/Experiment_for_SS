import re
import random

from sklearn import svm, metrics
import sklearn
# from sklearn.model_selection import cross_val_predic
from sklearn.model_selection import cross_val_predict, cross_val_score,ShuffleSplit
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# print(__doc__)
counterX = 0
folds  = 5
while counterX <= 10:
    counterX +=1
    feature_list = []
    class_list = []
    dict = []
    index_list = []

    i = 0
    count_for_0_class = 0
    total_1_class_instances = 2912
    feature_list_of_all_instances = []
    class_list_of_all_instances = []
    # limit_on_instances = 2912 * 2
    file_read = open('/home/farshid/Desktop/dataset.txt', 'r')

    count_for_number_of_instances = 0
    total_matrix = []
    while i < 300000:
        x = file_read.readline()
        if len(x) <= 10:
            break
        l = re.findall("-?\d+", x)
        l = list(map(int, l))
        total_matrix.append(l)
        feature_list_of_all_instances.append(l[0:1282])
        class_list_of_all_instances.append(l[1282])
        if l[1282] == 1:
            feature_list.append(l[0:1282])
            class_list.append(l[1282])
            count_for_0_class += 1


    # print(count_for_0_class)
    randomly_generated_indexes = []

    while(count_for_0_class + 10 > 0):
        randomly_generated_indexes.append(int(random.uniform(0,len(total_matrix))) )
        count_for_0_class -=1

    for cursor in randomly_generated_indexes:
        if total_matrix[cursor][1282] == 0:
            feature_list.append( total_matrix[cursor][0:1282])
            class_list.append( total_matrix[cursor][1282] )
        # feature_list_of_all_instances.append(l[0:1282])
        # class_list_of_all_instances.append(l[1282])

        # if l[1282] == 1:
        #     feature_list.append(l[0:1282])
        #     class_list.append(l[1282])
        #     # count_for_0_class += 1
        #     count_for_number_of_instances += 1
        #     index_list.append(i)
            # limit_on_instances -= 1
        # i += 1  # i = 0
    #
    # count_for_0_class = 0
    # total_1_class_instances = 2912
    # # limit_on_instances = 2912 * 2
    # file_read = open('/home/farshid/Desktop/dataset.txt', 'r')
    #
    # while count_for_number_of_instances > 0:
    #     x = file_read.readline()
    #     if len(x) <= 10:
    #         break
    #     l = re.findall("-?\d+", x)
    #     l = list(map(int, l))
    #
    #     if l[1282] == 0 and random.uniform(0, 1) > .4:
    #         feature_list.append(l[0:1282])
    #         class_list.append(l[1282])
    #         # count_for_0_class -= 1
    #         count_for_number_of_instances -= 1
    #         index_list.append(i)
    #         # limit_on_instances -= 1

    cou1 = 0
    cou2 = 0
    for h in class_list_of_all_instances:
        if h == 1:
            cou1 += 1
        if h == 0:
            cou2 += 1
    print("positive in list ")
    print(cou1)
    print("negetive in list ")
    print(cou2)
    # while j < i :
    #
    #     feature_list.append(dict[j][0:1282])
    #     class_list.append(dict[j][1282])
    #     j = j + 1
    #
    #
    #
    #
    #
    #
    #
    #

    feature_list = np.asarray(feature_list)
    class_list = np.asarray(class_list)
    feature_list = StandardScaler().fit_transform(feature_list)
    total_matrix = []

    feature_list_of_all_instances = feature_list_of_all_instances[:len(feature_list_of_all_instances)//2]
    feature_list_of_all_instances = StandardScaler().fit_transform(feature_list_of_all_instances)

    class_list_of_all_instances = class_list_of_all_instances[:len(class_list_of_all_instances) // 2]

    #
    clf = svm.SVC(max_iter=-1 , C=.8, kernel='rbf',degree=30, gamma='auto', cache_size=200)
    #
    # #
    #
    #
    clf.fit(feature_list, class_list)
    # #
    # # iris = datasets.load_iris()
    # # b = iris.data
    # # c = iris.target
    #
    #
    #
    # # predicted = cross_val_predict(clf, iris.data, iris.target, cv=5)
    cv = ShuffleSplit(n_splits=folds , test_size=.1,random_state=0)
    scores = cross_val_score(clf, feature_list_of_all_instances, class_list_of_all_instances, cv=folds)
    folds  += 5
    #
    # # print ( metrics.accuracy_score(iris.target, predicted) )
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # if max_mean < scores.mean():
    #     max_mean = scores.mean()

    # feature_list_of_all_instances = StandardScaler().fit_transform(feature_list_of_all_instances)

    ##########################################  scaling #######################################################

    # first_half_of_total_feature_list = feature_list_of_all_instances[:len(feature_list_of_all_instances)//2]
    #
    # first_half_of_total_feature_list = StandardScaler().fit_transform(first_half_of_total_feature_list)
    #
    # first_half_of_total_class_list = class_list_of_all_instances[:len(class_list_of_all_instances) // 2]

    ##########################################  scaling #######################################################

    # second_half_of_total_feature_list = feature_list_of_all_instances[len(feature_list_of_all_instances) / 2:]

    # predicted = cross_val_predict(clf, feature_list,class_list, cv=2)
    # metrics.accuracy_score(class_list, predicted)


    predicted = cross_val_predict(clf,feature_list_of_all_instances,class_list_of_all_instances,cv=folds)
    print("accurary for predictions " , end='')
    print(metrics.accuracy_score(class_list_of_all_instances,predicted))

    predicted = []
    j = 0
    for j in range(0, len(feature_list_of_all_instances)):
        val = [feature_list_of_all_instances[j]]
        predicted.append(clf.predict(val))

    # predicted.append( clf.predict([feature_list[1]]) )

    #   confusion matrix format
    #
    #   TN | FP
    #   FN | TP
    #
    #

    print("##############   confusion matrix and details  #############",end='\n\n')

    print("confusion matrix ")
    confusion_matrix = sklearn.metrics.confusion_matrix(class_list_of_all_instances, predicted)
    print(confusion_matrix)
    print("specificity " ,end='')
    specificity = float(confusion_matrix[0][0]) /( float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]) )
    print(specificity)
    print("sensitivity " ,end='')
    sensitivity = float( confusion_matrix[1][1] )/ float( (confusion_matrix[1][1]) + float(confusion_matrix[1][0]) )
    print(sensitivity)
    print("precision ",end='')
    precision = float( confusion_matrix[1][1]) / ( float(confusion_matrix[1][1]) + float(confusion_matrix[0][1] ) )
    print(precision)
    # fpr tpr threshold
    print("roc curve :- fpr, tpr , threshold ")
    print(sklearn.metrics.roc_curve(class_list_of_all_instances, predicted))

    # print("\n\n\n\n max mean", max_mean)
