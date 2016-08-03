#! /usr/bin/env python
import unittest
from DocClassifier import *
from pysam import FastaFile

class DocClassificationUnittest(unittest.TestCase):
    """Some functional tests.
    """
    testFile = './train_data/'#E114.bed'
    genome = FastaFile("./genome/GRCh38.genome.fa")

    def test_wrong_label(self):
        c = DocClassifier()
        print(self.testFile, self.genome)
        for label in c.extractDataPointRegionStringAndLabel(self.testFile, self.genome)[1]:
            self.assertTrue(label == PROMOTER or label == ENHANCER, "Error in training data labels: it must be one of \'promoter\' or \'enhancer\'")
        
    def test_wrong_region_size(self):
        c = DocClassifier()
        regions = c.extractDataPointRegionStringAndLabel(self.testFile, self.genome)[0]
        #print regions
        for region in regions:
            print "length: ", len(region)
            self.assertTrue(len(region) <= c.max_region_size, "Error extracting the region, min size of the region is not guaranteed")

    def test_right_string_content(self):
        c = DocClassifier()
        regions = c.extractDataPointRegionStringAndLabel(self.testFile, self.genome)[0]
        #print regions
        for region in regions:
            self.assertTrue('A' in region or "C" in region or "T" in region or "G" in region, "Wrong data format")

if __name__ == '__main__':
    unittest.main()