import pandas as pd
import numpy as np

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from sklearn.metrics import average_precision_score

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

print("Reading from text")
total_matrix = pd.read_csv('/home/farshid/Desktop/enzyme_dataset_1477.txt', header=None)
#
total_matrix.to_csv('/home/farshid/Desktop/decreased_1477.csv', header=False, index=False)

total_matrix['label'] = total_matrix[total_matrix.shape[1] - 1]
total_matrix.drop([total_matrix.shape[1] - 2], axis=1, inplace=True)

np.count_nonzero(total_matrix['label'] == -1)

features_of_all_instances = np.array(total_matrix.drop(['label'], axis=1))
class_of_all_instances = np.array(total_matrix['label'])

class_of_all_instances = LabelEncoder().fit_transform(class_of_all_instances)

features_of_all_instances = Normalizer().fit_transform(features_of_all_instances)

sampler = RandomUnderSampler(ratio=.99)
sampler = SMOTE()

all_runs_average_auc = []
all_runs_average_aupr = []

for i in range(0, 6):

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    all_auc = []
    all_aupr = []

    for train_index, test_index in skf.split(features_of_all_instances, class_of_all_instances):
        #        X_train = new_X[train_index]
        #        X_test = new_X[test_index]

        X_train = features_of_all_instances[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        #    X_train, y_train = tl.fit_sample(X_train, y_train)
        #    X_train, y_train = smote.fit_sample(X_train, y_train)

        #    X_train, y_train = enn.fit_sample(X_train, y_train)

        #        X_train , y_train = RandomUnderSampler().fit_sample(X_train,y_train)

        X_sampled, y_sampled = under_sampler.fit_sample(X_train, y_train)
        #    X_train , y_train = oversampler.fit_sample(X_train,y_train)

        best_classifier.fit(X_sampled, y_sampled)

        #    X_sampled , y_sampled = sampler.fit_sample(X_train,y_train)

        #    len_before_over = len(y_sampled)

        #    X_sampled , y_sampled = sampler.fit_sample(X_sampled,y_sampled)

        #    len_after_over = len(y_sampled)

        #    data_increases = len_after_over - len_before_over

        #    np.concatenate((X_train,X_sampled[-data_increases:,:]),axis=0)
        #    np.concatenate((y_train,y_sampled[-data_increases:]),axis=0)

        #    custom_classifier.fit(X_train, y_train)

        # for custom adaboost
        #    predictions , ignored = custom_classifier.predict(X_test)

        # for built in adaboost
        predictions = best_classifier.predict_proba(X_test)

        #    print(predictions)

        all_auc.append(roc_auc_score(y_test, predictions[:, 1]))

        #        all_auc.append(roc_auc_score(y_test, best_classifier.predict(X_test)))
        #
        #    fpr, tpr, threshold = roc_curve(y_test, predictions)
        #    all_auc.append(auc(fpr,tpr))


        all_aupr.append(average_precision_score(y_test, predictions[:, 1]))

        #        print(classification_report(y_test,predictions))

        print('1 fold done')

    average_auc = sum(all_auc) / len(all_auc)
    average_aupr = sum(all_aupr) / len(all_aupr)

    all_runs_average_auc.append(average_auc)

    break

# all_runs_average_aupr.append(average_aupr)

auc_all_runs_average = sum(all_runs_average_auc) / len(all_runs_average_auc)
aupr_all_runs_average = sum(all_runs_average_aupr) / len(all_runs_average_aupr)
