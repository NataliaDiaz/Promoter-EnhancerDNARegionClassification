# Promoter/ Enhancer DNA Region Classification
Binary classification problem for DNA region strings

###Brief Background
In genetics, an enhancer is a short (100-1000 base pairs) region of DNA that can be bound by proteins (activators) to increase the likelihood transcription will occur at a gene. Promoter is a similar sized region of the DNA that initiates transcription of a particular gene.

###Problem
Given a DNA region of 1000 base pairs, predict if it is an enhancer or a promoter. In other words, given a string with 1000 characters (A/C/G/T) you have to classify it into one of two possible classes based on its sequence, classify a genomic region as either a promoter or an enhancer (binary classification).

### Notes:
* It may be helpful to think of the genome as a document written in an unknown language where words are allowed to overlap
* The simplest way to run the helper code is to install anaconda python v2, and then pip install pysam
* The regions are of variable lengths, but the program extract_regions_example.py gets and centers the regions in a 1000 bp window.
* Data (you may need to subsample the data): https://www.dropbox.com/s/8cfvxguuf2qzf9w/data_science_programming_challenge.tar?dl=0


-------------
### Instructions to run the program:

#### Program Solution 1: 
Using Random Forest Classifier and pandas dummy variables for tackling missing data (i.e. different size input strings) in a way that every character in the dna string is a feature with 4 possible values ('A','C','T','G') encoded in such a way that its occurrence is modelled with (1) and otherwise (0).

`$py PromoterDiscoveryGenomeClassificationRF.py` 

```python
##### Example of output:
Registered  27500  regions and  27500  labels
Unique region values:  24817
Creating model...
Cross-validating...
Score is...
10-fold Train CV | Accuracy: 0.721 +/- 0.0
```

-------------
#### Program Solution 2: 
Benchmark using Ridge Classifier, Perceptron, Passive-Aggressive, kNN, Random forest. This method uses a sparse tf-idf vectorizer to extract features (taking the string values as part of a language vocabulary), and a chi-squared test to select the best performing features.

`$py PromoterDiscoveryGenomeClassificationDocClassification.py` 

##### Example of output (best result achieved):
```python
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
train time: 76.110s
test time:  12.167s
accuracy:   0.731

training times:  [  1.13212160e-03   1.20041741e-03   1.33371359e-03   9.37728209e-05
   1.00000000e+00]
test_times:  [  3.05685348e-05   3.46051490e-05   2.67082775e-05   4.60738087e-01
   1.00000000e+00]```


-------------
### To run the unittest: 

`$py tests/unittest.py`
