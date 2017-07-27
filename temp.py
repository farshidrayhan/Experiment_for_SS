################################################################################
#filenames = tf.train.match_filenames_once('C:/Users/Farshid/Desktop/datasets/png/train/')
#import tensorflow as tf
#from skimage import io
#import matplotlib.image as img
#filename_queue = tf.train.string_input_producer(
#tf.train.match_filenames_once("C:/Users/Farshid/Desktop/datasets/png/labeled/*/*.png"))


#import os
#from os.path import join, getsize
#x= []
#y = []
#
#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#    
#for root, dirs, files in os.walk('C:/Users/Farshid/Desktop/datasets/'):
#    
#    
##    print(root," ",dirs,  " " , files )
###################################################################################    
#    if dirs == [] :
##        print("path ",root," label ",root[len(root)-1]," found " , files )
#        
#        for file in files:
#            file_path = root[:len(root)-1] + '' + root[len(root)-1] + '/' + file
#            c = io.imread(file_path,as_grey=True)
##            img.imread('C:/Users/Farshid/Desktop/datasets/1/1.png')
#            x.append(c)
#            y.append(root[len(root)-1])
#
#for i in range(len(x)):
#    print(x[i].shape)         
    
###############################################################################    
#        print("found " , files, " in " , dirs , "directory")
#    
#    print(sum(getsize(join(root, name)) for name in files))
#    print("bytes in", len(files), "non-directory files" )
#    if 'CVS' in dirs:
#        dirs.remove('CVS')  # don't visit CVS directories



#
#import tensorflow as tf
#filename_queue = tf.train.string_input_producer(
#tf.train.match_filenames_once("C:/Users/Farshid/Desktop/datasets/png/labeled/*/*.png"))
#
#image_reader = tf.WholeFileReader()
#key, image_file = image_reader.read(filename_queue)
#S = tf.string_split([key],'/')
#length = tf.cast(S.dense_shape[1],tf.int32)
## adjust constant value corresponding to your paths if you face issues. It should work for above format.
#label = S.values[length-tf.constant(2,dtype=tf.int32)]
#label = tf.string_to_number(label,out_type=tf.int32)
#image = tf.image.decode_png(image_file)
#
## Start a new session to show example output.
#with tf.Session() as sess:
#    # Required to get the filename matching to run.
##    tf.initialize_all_variables().run()
#    tf.global_variables_initializer()
#    
#
#    # Coordinate the loading of image files.
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    for i in range(8):
#        key_val,label_val,image_tensor = sess.run([key,key,key])
#        print(image_tensor.shape)
#        print(key_val)
#        print(label_val)
#
#
#   # Finish off the filename queue coordinator.
#coord.request_stop()
#coord.join(threads)
#






"""
Created on Wed May 17 00:43:51 2017

@author: Farshid
"""

from multiprocessing import Pool
import math
from math import factorial

import os
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

#datasets = ['nr_dataset_1477.txt', 'nr_dataset_1404.txt', 'nr_dataset_1294.txt', 'nr_dataset_1282.txt',
#            'gpcr_dataset_1477.txt', 'gpcr_dataset_1404.txt', 'gpcr_dataset_1294.txt', 'gpcr_dataset_1282.txt']
            #'enzyme_dataset_1477.txt', 'enzyme_dataset_1404.txt', 'enzyme_dataset_1294.txt', 'enzyme_dataset_1282.txt']

datasets = ['ir_dataset_1282.txt','ir_dataset_1477.txt','ir_dataset_1404.txt','ir_dataset_1294.txt']


def create_model(dataset):
    print("dataset : ", dataset)
    df = pd.read_csv('/home/farshid/Desktop/' + dataset, header=None)

    df['label'] = df[df.shape[1] - 1]

    df.drop([df.shape[1] - 2], axis=1, inplace=True)

    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])

    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])

    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)

    # This part is for stratified cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # This part is for Random Undersampling
    sampler = RandomUnderSampler()

    top_roc = 0
    for depth in range(2, 20,1):
        for split in range(2, 9,1):

            all_auc = []
            all_aupr = []

            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=depth, min_samples_split=split),
                n_estimators=100,
                learning_rate=1, algorithm='SAMME')

            for train_index, test_index in skf.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index]

                X_train, y_train = sampler.fit_sample(X_train, y_train)

                classifier.fit(X_train, y_train)

                predictions = classifier.predict_proba(X_test)

                all_auc.append(roc_auc_score(y_test, predictions[:, 1]))
                all_aupr.append(average_precision_score(y_test, predictions[:, 1]))

            average_auc = sum(all_auc) / len(all_auc)
            average_aupr = sum(all_aupr) / len(all_aupr)
            
            # print("for depth", depth, " and split ", split, "roc = ", average_auc)
            if average_auc > top_roc:
                print(dataset, " for depth", depth, " split ", split, "roc = ", average_auc," aupr ", average_aupr, end=' ')
                joblib.dump(classifier, '/home/farshid/Desktop/models/' + dataset + '.pkl')
                top_roc = average_auc
                print("stored !!!!")


def func(x):
    #    print('process id:', os.getpid() )

    
    # math.factorial(x)
    create_model(x)
    

if __name__ == '__main__':
    input('Enter any Key to start ')
    pool = Pool(4)
    results = pool.map(func, datasets)
	
    input('Enter any Key to end ')
#

















#
#image_reader = tf.WholeFileReader()
#_, image_file = image_reader.read(filename_queue)
#image = tf.image.decode_png(image_file)
#
## Start a new session to show example output.
#with tf.Session() as sess:
#    # Required to get the filename matching to run.
#    tf.initialize_all_variables().run()
#
#    # Coordinate the loading of image files.
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    for i in range(6):
#        # Get an image tensor and print its value.
#        image_tensor = sess.run(image)
#        print(image_tensor.shape)
#
#   # Finish off the filename queue coordinator.
#    coord.request_stop()
#    coord.join(threads)
#    