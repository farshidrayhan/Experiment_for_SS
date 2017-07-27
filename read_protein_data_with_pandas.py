import pandas as pd
import numpy as np

dataset = pd.read_csv('C:\\Protein Thesis\\feature_reduced_dataset.txt', sep=",", header = None)

dataset = dataset.as_matrix(columns=None)


num_of_columns = dataset.shape[1]
num_of_rows = dataset.shape[0]

X = dataset[:, 0:num_of_columns - 1]
y = dataset[:, num_of_columns - 1]

from sklearn.preprocessing import Normalizer


normalization_object = Normalizer()
X = normalization_object.fit_transform(X)

import sklearn.svm as svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

from sklearn.metrics import average_precision_score


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.over_sampling import SMOTE

enn = EditedNearestNeighbours(size_ngh=5,n_jobs=3, kind_sel='all')
smote = SMOTE(n_jobs=3)
X_resampled, y_resampled = enn.fit_sample(X, y)

# Apply Tomek Links cleaning
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_sample(X, y)

clf = DecisionTreeClassifier(max_depth=15)

from sklearn.metrics import classification_report

y[y==0] = -1

classifier = AdaBoostClassifier(
    clf,
    n_estimators=150,
    learning_rate=1,algorithm='SAMME')

import sys
sys.path.append("C:\\Protein Thesis")

from rusboost import AdaBoost

custom_classifier = AdaBoost(500,depth=50)


import sys
sys.path.append("C:\\Protein Thesis")

from smoteboost import AdaBoost

custom_classifier = AdaBoost(100,depth=10)


# This part is for stratified cross validation
skf = StratifiedKFold(n_splits=5,shuffle=True)

# This part is for Random Undersampling
sampler = RandomUnderSampler(replacement=False)
oversampler = SMOTE()

from imblearn.under_sampling import NeighbourhoodCleaningRule

ncr = NeighbourhoodCleaningRule(size_ngh=5)

#classifier = svm.SVC(C=400)

all_auc = []
all_aupr = []



for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]


    
#    X_train, y_train = tl.fit_sample(X_train, y_train)
#    X_train, y_train = smote.fit_sample(X_train, y_train)

#    for i in range(7) :
#        X_train, y_train = ncr.fit_sample(X_train, y_train)

    X_train , y_train = sampler.fit_sample(X_train,y_train)
#    X_train , y_train = oversampler.fit_sample(X_train,y_train)
    
    
    
    
    
    classifier.fit(X_train, y_train)

#    X_sampled , y_sampled = sampler.fit_sample(X_train,y_train)
    
#    len_before_over = len(y_sampled)
    
#    X_sampled , y_sampled = oversampler.fit_sample(X_train,y_train)
    
#    len_after_over = len(y_sampled)
    
#    data_increases = len_after_over - len_before_over
    
#    np.concatenate((X_train,X_sampled[-data_increases:,:]),axis=0)
#    np.concatenate((y_train,y_sampled[-data_increases:]),axis=0)
    
#    custom_classifier.fit(X_train, y_train)
    
    #for custom adaboost
#    predictions , ignored = custom_classifier.predict(X_test)
    
    #for built in adaboost
    predictions  = classifier.predict(X_test)
    
#    print(predictions)

#    classifier.fit(X_sampled,y_sampled)
    
#    predictions = classifier.predict(X_test)
    
    all_auc.append(roc_auc_score(y_test, predictions))
#
#    fpr, tpr, threshold = roc_curve(y_test, predictions)
#    all_auc.append(auc(fpr,tpr))
    
    
    all_aupr.append(average_precision_score(y_test, predictions))
    
    print(classification_report(y_test,predictions))
    
    print('1 fold done')
    
average_auc = sum(all_auc)/len(all_auc)
average_aupr = sum(all_aupr)/len(all_aupr)




x1,y1 , ind1 = sampler.fit_sample(X,y)
x2,y2 , ind2 = sampler.fit_sample(X,y)


array_one = np.array([0,1,2,3,4,5,6,7])

array_two = np.array([0,2,4])

array_three = np.concatenate((array_one,array_two),axis=0)







#nr dataset experiment begins

import pandas as pd
import numpy as np

dataset = pd.read_csv('C:\\Protein Thesis\\nr_dataset_feature_reduced.txt', sep=",", header = None)

dataset = dataset.as_matrix(columns=None)


num_of_columns = dataset.shape[1]
num_of_rows = dataset.shape[0]

X = dataset[:, 0:num_of_columns - 1]
y = dataset[:, num_of_columns - 1]

from sklearn.preprocessing import Normalizer


normalization_object = Normalizer()
X = normalization_object.fit_transform(X)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import average_precision_score


from sklearn.metrics import classification_report

y[y==0] = -1

import sys
sys.path.append("C:\\Protein Thesis")

from rusboost import AdaBoost

custom_classifier = AdaBoost(200,depth=15)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=15)



classifier = AdaBoostClassifier(
    clf,
    n_estimators=100,
    learning_rate=1,algorithm='SAMME')

# This part is for stratified cross validation
skf = StratifiedKFold(n_splits=10,shuffle=True)


all_auc = []
all_aupr = []



for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]


    
#    X_train, y_train = tl.fit_sample(X_train, y_train)
#    X_train, y_train = smote.fit_sample(X_train, y_train)
    
#    X_train, y_train = enn.fit_sample(X_train, y_train)

#    X_train , y_train = sampler.fit_sample(X_train,y_train)
#    X_train , y_train = oversampler.fit_sample(X_train,y_train)
    
#    classifier.fit(X_train, y_train)

#    X_sampled , y_sampled = sampler.fit_sample(X_train,y_train)
    
#    len_before_over = len(y_sampled)
    
#    X_sampled , y_sampled = sampler.fit_sample(X_sampled,y_sampled)
    
#    len_after_over = len(y_sampled)
    
#    data_increases = len_after_over - len_before_over
    
#    np.concatenate((X_train,X_sampled[-data_increases:,:]),axis=0)
#    np.concatenate((y_train,y_sampled[-data_increases:]),axis=0)
    
    custom_classifier.fit(X_train, y_train)
    
    #for custom adaboost
    predictions , ignored = custom_classifier.predict(X_test)
    
    #for built in adaboost
#    predictions  = classifier.predict(X_test)
    
#    print(predictions)
    
    all_auc.append(roc_auc_score(y_test, predictions))
#
#    fpr, tpr, threshold = roc_curve(y_test, predictions)
#    all_auc.append(auc(fpr,tpr))
    
    
    all_aupr.append(average_precision_score(y_test, predictions))
    
    print(classification_report(y_test,predictions))
    
    print('1 fold done')
    
average_auc = sum(all_auc)/len(all_auc)
average_aupr = sum(all_aupr)/len(all_aupr)



