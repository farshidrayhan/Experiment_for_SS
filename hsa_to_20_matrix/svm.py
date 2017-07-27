from sklearn import svm, metrics

from sklearn.model_selection import cross_val_predict,cross_val_score
clf = svm.SVC()
#

feature_list = []
feature_list.append([0,0,0 ])
feature_list.append([0,0,1 ])
feature_list.append([0,1,0 ])
feature_list.append([0,1,1 ])
feature_list.append([1,0,0 ])
feature_list.append([1,0,1 ])
feature_list.append([1,1,0 ])
feature_list.append([1,1,1 ])


class_list = [0,0,0,1,0,0,0,1 ]


clf.fit(feature_list,class_list)
scores = cross_val_score(clf, feature_list, class_list, cv=2)
# predicted = cross_val_predict(clf, feature_list, class_list, cv=2)
# metrics.accuracy_score(class_list, predicted)
# print ( metrics.accuracy_score(feature_list, class_list) )
# accuracy = clf.score(test_matrix,ALL_train)
# print (accuracy)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# clf.class_weight(feature_list[1],2)
# print(len(feature_list[0]))

# str = class_list[0]
# my_list = str.split(",")
# print(len(my_list))


# print(my_list)

# print(len(class_list[0]))


print (clf.predict(feature_list) )


