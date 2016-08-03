#! /usr/bin/env python
import sys
import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm
import math
from sys import exit
#import pudb # trace debugger
import os
import os.path
#from DataframeUtils import *
#from NLPUtils import *
#from SQLUtils import *
import scipy.stats as stats
import matplotlib.pyplot as plt  #%matplotlib inline
import seaborn as sns
import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os, sys
from pysam import FastaFile
# Compute confusion matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
# Standardize features by removing the mean and scaling to unit variance
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
# Grid Search Random Forest parameters
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

min_region_size = 1000
genomeDirectory = './genome/'
dataDirectory = './train_data/'
genome = FastaFile("./genome/GRCh38.genome.fa")
dataFiles = ['E114.bed', 'E116.bed', 'E117.bed', 'E118.bed', 'E119.bed']#, 'E120.bed', 'E121.bed', 'E122.bed', 'E123.bed', 'E124.bed', 'E126.bed', 'E127.bed', 'E128.bed', 'E129.bed']
c = 0
regions = []
labels = []

def iter_peaks_and_labels(fname):
    with open(fname) as fp:
        for line in fp:
            data = line.split()
            yield (data[0], int(data[1]), int(data[2])), data[3]   # returns region and its label: ('chrY', 20575266, 20576266),   'promoter'/'enhancer'
    return

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Function used to print cross-validation scores
def training_score(est, X, y, cv):
    acc = cross_val_score(est, X, y, cv = cv, scoring='accuracy')
    #roc = cross_val_score(est, X, y, cv = cv, scoring='roc_auc')
    print '10-fold Train CV | Accuracy:', round(np.mean(acc), 3),'+/-', round(np.std(acc), 3)
    #,'| ROC AUC:', round(np.mean(roc), 3), '+/-', round(np.std(roc), 3)

if __name__ == '__main__':
    min_region_size = 1000
    genomeDirectory = './genome/'
    dataDirectory = './train_data/'
    genome = FastaFile("./genome/GRCh38.genome.fa")
    dataFiles = ['E114.bed']#, 'E116.bed']#, 'E117.bed', 'E118.bed', 'E119.bed']#, 'E120.bed', 'E121.bed', 'E122.bed', 'E123.bed', 'E124.bed', 'E126.bed', 'E127.bed', 'E128.bed', 'E129.bed']
    c = 0
    regions = []
    labels = []    
    for i in range(len(dataFiles)):
        for region, label in iter_peaks_and_labels(dataDirectory+dataFiles[i]):#sys.argv[1]):
            # create a new region exactly min_region_size basepairs long centered on 
            # region  
            expanded_start = region[1] + (region[2] - region[1])/2 - min_region_size/2
            expanded_stop = expanded_start + min_region_size
            #region = (region[0], expanded_start, expanded_stop)    
            #print "Region: ", region, " Label: ", label
            #allRegions.append(Region(region, genome.fetch(*region), label))
            #dataPoints.append((genome.fetch(*region), label))
            regions.append(genome.fetch(*region))
            if label == 'promoter':  # = 1
                labels.append(1)
            elif label == 'enhancer':
                labels.append(0)
            else:
                print "Wrong label"
                sys.exit(-1)
    #         if len(regions)>10:
    #             break
    #         c += 1
    print "Registered ", len(regions)," regions and ", len(labels), " labels"
    #data_df = pd.DataFrame({'region' : regions, 'label' : labels}, columns=["region", "label"])
    train_df = pd.DataFrame({'region' : regions, 'label' : labels}, columns=["region", "label"])
    train_df.describe()
    print "Unique region values: ", train_df.region.nunique()
    #train_df
    #print " header \n", train_df.head()
    
    testFiles = ['E114.bed']#23.bed']#, 'E124.bed']#, 'E126.bed']
    regions = []
    labels = []
    for i in range(len(testFiles)):
        for region, label in iter_peaks_and_labels(dataDirectory+dataFiles[i]):#sys.argv[1]):
            # create a new region exactly min_region_size basepairs long centered on 
            # region  
            expanded_start = region[1] + (region[2] - region[1])/2 - min_region_size/2
            expanded_stop = expanded_start + min_region_size
            #region = (region[0], expanded_start, expanded_stop)    
            #print "Region: ", region, " Label: ", label
            #allRegions.append(Region(region, genome.fetch(*region), label))
            #dataPoints.append((genome.fetch(*region), label))
            regions.append(genome.fetch(*region))
            if label == 'promoter':  # = 1
                labels.append(1)
            elif label == 'enhancer':
                labels.append(0)
            else:
                print "Wrong label"
                sys.exit(-1)
    test_df = pd.DataFrame({'region' : regions, 'label' : labels}, columns=["region", "label"])


    # Featurize the data
    # convert the "label" label column to numpy arrays
    train_converted = train_df.pop('label').values
    test_converted = test_df.pop('label').values

    # transform the categorical features to binary features
    train_dummies_df = pd.get_dummies(train_df)
    test_dummies_df = pd.get_dummies(test_df)

    # get the feature names - this will be useful for the model visualization and feature analysis
    features = train_dummies_df.columns.values
    #print "Features: ", features  #  [ 'region_AAAAAAAAAAAAAAGAAAAAAAACCCCGCCGGAT', ...

    # convert the training and test dataframes to numpy arrays
    train_data = train_dummies_df.values
    test_data = test_dummies_df.values

    # print 'training data shape', train_data.shape
    # print 'test data shape', test_data.shape
    # print 'converted label data shape', train_converted.shape
    # print features

    #Split the data into a training set and a test set
    #X_train, X_test, Y_train, Y_test = train_test_split(X, y)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    #  Model tuning:
    X = train_scaled
    y = train_converted
    print " train X scaled: ", X
    print " train y converted: ", y

    # Build model
    ###################### Initial example
    print "Creating model..."
    rfc = RandomForestClassifier(criterion='entropy', max_features=None, max_depth=None,#3, #“gini” for the Gini impurity and “entropy” for the information gain
                                 min_samples_split = 10, min_samples_leaf=5,
                                 n_estimators = 100,
                                 n_jobs=-1)


    ###################### Example A
    # # Perform a grid search on random forest parameters
    # random_forest_grid = {'max_depth': [3, 5, None],
    #                       'max_features': ['sqrt', 'log2', None],  #1, 3, 'auto'
    #                       'min_samples_split': [10],
    #                       'min_samples_leaf': [5],
    #                       'n_estimators': [100], #10
    #                       'random_state': [1]}
    # rf_gridsearch = GridSearchCV(RandomForestClassifier(),
    #                              random_forest_grid,
    #                              n_jobs=-1,
    #                              verbose=True,
    #                              scoring='accuracy')
    # rf_gridsearch.fit(X, y)
    # print "Best parameters:", rf_gridsearch.best_params_
    # best_rf_model = rf_gridsearch.best_estimator_s
    

    # Example B 
    # Reddit comments: RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #         max_depth=None, max_features='auto', max_leaf_nodes=None,
    #         min_samples_leaf=1, min_samples_split=2,
    #         min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
    #         oob_score=False, random_state=None, verbose=0,
    #         warm_start=False)
    # train time: 3.338s
    # test time:  0.145s
    # accuracy:   0.842

    # Example C
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # dataframe = pandas.read_csv(url, names=names)
    # array = dataframe.values
    # X = array[:,0:8]
    # Y = array[:,8]
    # num_folds = 10
    # num_instances = len(X)
    # seed = 7
    # num_trees = 100
    # max_features = 3
    # kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    # results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    # print(results.mean())

    print "Cross-validating..."
    # Cross Validate the best model
    cv = StratifiedKFold(y, n_folds=10, shuffle=True)
    print "Score is..."
    score = training_score(rfc, X, y, cv)   # 0.721 acc avg
    #score = training_score(best_rf_model, X, y, cv)   # memory crash Killed 9




    