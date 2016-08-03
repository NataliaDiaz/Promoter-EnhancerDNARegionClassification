#! /usr/bin/env python
from __future__ import print_function  # requires print() format

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import os, sys
from pysam import FastaFile

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import train_test_split

# Labels
PROMOTER = 0 #'promoter'
ENHANCER = 1 #'enhancer'

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


class DocClassifier(object):
  max_region_size = 1000

  def iter_peaks_and_labels(self, fname):
    with open(fname) as fp:
        for line in fp:
            data = line.split()
            yield (data[0], int(data[1]), int(data[2])), data[3]   # returns region and its label: ('chrY', 20575266, 20576266),   'promoter'/'enhancer'
    return

  def extractDataPointRegionStringAndLabel(self, directory, genome):
    ''' Returns a list of data points in form of an array of dna strings and an array of labels for each datapoint'''
    dataFiles = os.listdir(directory)
    files = []
    for s in dataFiles:
      if ".bed" in s:
        files.append(s) 
    print("Input DataFiles: ", files)
    regions = []
    labels = []
    #### TODO: UNCOMMENT EXTERNAL FOLLOWING LOOP TO RUN OVER ALL FILES (Takes longer)
    #for i in files: 
      #print ("Using training file: ", filename)
    inputTestFile = directory+files[0]   
    print ("Using test file: ", inputTestFile, " Use full dataDirectory for complete training")
    for region, label in self.iter_peaks_and_labels(inputTestFile):#sys.argv[1]):
        # create a new region exactly max_region_size basepairs long centered on 
        # region  
        expanded_start = region[1] + (region[2] - region[1])/2 - self.max_region_size/2
        expanded_stop = expanded_start + self.max_region_size
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
          print ("Wrong label: ", label)
          sys.exit(-1)
    return regions, labels
  
  ###############################################################################
  # Benchmark classifiers
  def benchmark(self, clf):
      print('_' * 80)
      print("Training: ")
      print(clf)
      t0 = time()
      clf.fit(X_train, y_train)
      train_time = time() - t0
      print("train time: %0.3fs" % train_time)

      t0 = time()
      pred = clf.predict(X_test)
      test_time = time() - t0
      print("test time:  %0.3fs" % test_time)

      score = metrics.accuracy_score(y_test, pred)
      print("accuracy:   %0.3f" % score)

      if hasattr(clf, 'coef_'):
          print("dimensionality: %d" % clf.coef_.shape[1])
          print("density: %f" % density(clf.coef_))

          if opts.print_top10 and feature_names is not None:
              print("top 10 keywords per class:")
              for i, category in enumerate(categories):
                  top10 = np.argsort(clf.coef_[i])[-10:]
                  print(trim("%s: %s"
                        % (category, " ".join(feature_names[top10]))))
          print()

      if opts.print_report:
          print("classification report:")
          print(metrics.classification_report(y_test, pred,
                                              target_names=categories))

      if opts.print_cm:
          print("confusion matrix:")
          print(metrics.confusion_matrix(y_test, pred))

      print()
      clf_descr = str(clf).split('(')[0]
      return clf_descr, score, train_time, test_time


if __name__ == '__main__':
  c = DocClassifier()
  # Display progress logs on stdout
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)s %(message)s')

  # parse commandline arguments
  op = OptionParser()
  op.add_option("--report",
                action="store_true", dest="print_report",
                help="Print a detailed classification report.")
  op.add_option("--chi2_select",
                action="store", type="int", dest="select_chi2",
                help="Select some number of features using a chi-squared test")
  op.add_option("--confusion_matrix",
                action="store_true", dest="print_cm",
                help="Print the confusion matrix.")
  op.add_option("--top10",
                action="store_true", dest="print_top10",
                help="Print ten most discriminative terms per class"
                     " for every classifier.")
  op.add_option("--all_categories",
                action="store_true", dest="all_categories",
                help="Whether to use all categories or not.")
  op.add_option("--use_hashing",
                action="store_true",
                help="Use a hashing vectorizer.")
  op.add_option("--n_features",
                action="store", type=int, default=2 ** 16,
                help="n_features when using the hashing vectorizer.")
  op.add_option("--filtered",
                action="store_true",
                help="Remove newsgroup information that is easily overfit: "
                     "headers, signatures, and quoting.")

  (opts, args) = op.parse_args()
  if len(args) > 0:
      op.error("this script takes no arguments.")
      sys.exit(1)

  print(__doc__)
  op.print_help()
  print()


  ###############################################################################
  # Load data

  genomeDirectory = './genome/'
  dataDirectory = './train_data/' #dataFiles = ['E114.bed', 'E116.bed']#, 'E117.bed', 'E118.bed', 'E119.bed']#, 'E120.bed', 'E121.bed', 'E122.bed', 'E123.bed', 'E124.bed', 'E126.bed', 'E127.bed', 'E128.bed', 'E129.bed'] 
  genome = FastaFile("./genome/GRCh38.genome.fa")
  
  regions, labels = c.extractDataPointRegionStringAndLabel(dataDirectory, genome)

  # split a training set and a test set
  X_train, X_test, y_train, y_test = train_test_split(regions, labels, test_size=0.33, random_state=42)

  print ("Registered ", len(regions)," regions and ", len(labels), " labels")
  # test_df = pd.DataFrame({'region' : regions, 'label' : labels}, columns=["region", "label"])
  # train_df = pd.DataFrame({'region' : regions, 'label' : labels}, columns=["region", "label"])
  # train_df.describe()
  # print "Unique region values: ", train_df.region.nunique()

  categories = set(labels) #train_df.region.nunique() #data_train.target_names    # for case categories == None
  print('Data loaded. Categories: %s', categories)

  data_train_size_mb = size_mb(X_train)
  data_test_size_mb = size_mb(X_test)

  print("%d documents train " % len(X_train))
  print("%d documents test " % len(X_test))
  print("%d categories train " % len(y_train))
  print("%d categories test " % len(y_test))


  print("Extracting features from the training data using a sparse vectorizer")
  t0 = time()
  if opts.use_hashing:
      vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                     n_features=opts.n_features)
      X_train = vectorizer.transform(X_train)
  else:
      vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                   stop_words='english')
      X_train = vectorizer.fit_transform(X_train)
  duration = time() - t0
  print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
  print("n_samples: %d, n_features: %d" % X_train.shape)
  print()

  print("Extracting features from the test data using the same vectorizer")
  t0 = time()
  X_test = vectorizer.transform(X_test)
  duration = time() - t0
  print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
  print("n_samples: %d, n_features: %d" % X_test.shape)
  print()

  # mapping from integer feature name to original token string
  if opts.use_hashing:
      feature_names = None
  else:
      feature_names = vectorizer.get_feature_names()
  #print ("feature_names: ",feature_names)

  if opts.select_chi2:
      print("Extracting %d best features by a chi-squared test" %  opts.select_chi2)
      t0 = time()
      ch2 = SelectKBest(chi2, k=opts.select_chi2)
      X_train = ch2.fit_transform(X_train, y_train)
      X_test = ch2.transform(X_test)
      if feature_names:
          # keep selected feature names
          feature_names = [feature_names[i] for i
                           in ch2.get_support(indices=True)]
      print("done in %fs" % (time() - t0))

  if feature_names:
      feature_names = np.asarray(feature_names)
      print ("Feature names: ", feature_names)

  results = []
  for clf, name in (
          (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
          (Perceptron(n_iter=50), "Perceptron"),
          (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
          (KNeighborsClassifier(n_neighbors=10), "kNN"),
          (RandomForestClassifier(n_estimators=100), "Random forest")):  # criterion='gini' for the Gini impurity and 'entropy' for the information gain 
      print('=' * 80)
      print(name)
      results.append(c.benchmark(clf))

  indices = np.arange(len(results))
  results = [[x[i] for x in results] for i in range(4)]

  clf_names, score, training_time, test_time = results
  training_time = np.array(training_time) / np.max(training_time)
  test_time = np.array(test_time) / np.max(test_time)

  print ("training times: ", training_time)
  print ("test_times: ", test_time)
