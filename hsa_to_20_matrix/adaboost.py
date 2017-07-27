import pandas as pd
import numpy as np
from sklearn.ensemble.tests.test_forest import check_min_samples_split

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

print("Reading ...")
df = pd.read_csv('/home/farshid/Desktop/decreased_1477.csv', header = None,)

# df.to_csv('/home/farshid/Desktop/decreased_1477.csv',header=False,index=False)
print("Arranging ")

df['label'] = df[df.shape[1]-1]
df.drop([df.shape[1]-2],axis=1,inplace=True)

np.count_nonzero(df['label'] == -1)

X = np.array(df.drop(['label'],axis=1))
y = np.array(df['label'])

y = LabelEncoder().fit_transform(y)

normalizer = Normalizer()

X = normalizer.fit_transform(X)



under_sampler = RandomUnderSampler()

over_sampler = SMOTE()

#trying to incorporate gridsearch


# X , y = under_sampler.fit_sample(X,y)


# grid_search = grid_search.fit(X, y)
#
# best_accuracy = grid_search.best_score_
#
# best_parameters = grid_search.best_params_
#
# best_classifier = grid_search.best_estimator_


skf = StratifiedKFold(n_splits=5,shuffle=True)


all_runs_average_auc = []
all_runs_average_aupr = []

for i in range(1,6) :

    skf = StratifiedKFold(n_splits=5,shuffle=True)

    best_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=13+i, min_samples_split=i,
                                                            min_samples_leaf= 2                    ),
                                         n_estimators=100, algorithm='SAMME')

    print(" i  = ",i)

    all_auc = []
    all_aupr = []
    
    for train_index, test_index in skf.split(X, y):
#        X_train = new_X[train_index]
#        X_test = new_X[test_index]
    
        X_train = X[train_index]
        X_test = X[test_index]
    
        y_train = y[train_index]
        y_test = y[test_index]
    
    
        
    #    X_train, y_train = tl.fit_sample(X_train, y_train)
    #    X_train, y_train = smote.fit_sample(X_train, y_train)
        
    #    X_train, y_train = enn.fit_sample(X_train, y_train)
    
#        X_train , y_train = RandomUnderSampler().fit_sample(X_train,y_train)
        
        X_sampled , y_sampled = under_sampler.fit_sample(X_train,y_train)
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
        
        #for custom adaboost
    #    predictions , ignored = custom_classifier.predict(X_test)
        
        #for built in adaboost
        predictions  = best_classifier.predict_proba(X_test)
        
    #    print(predictions)
        print("roc :- " , roc_auc_score(y_test, predictions[:,1]))
        all_auc.append(roc_auc_score(y_test, predictions[:,1]))
        
#        all_auc.append(roc_auc_score(y_test, best_classifier.predict(X_test)))
    #
    #    fpr, tpr, threshold = roc_curve(y_test, predictions)
    #    all_auc.append(auc(fpr,tpr))
        

        
#        print(classification_report(y_test,predictions))
        
        print('1 fold done')
        
    average_auc = sum(all_auc)/len(all_auc)
    print(average_auc)
    
    all_runs_average_auc.append(average_auc)
    
    # break
    
#    all_runs_average_aupr.append(average_aupr)
    
print(sum(all_runs_average_auc)/len(all_runs_average_auc) )
        