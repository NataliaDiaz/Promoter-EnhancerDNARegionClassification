#! /usr/bin/env python
import os, sys
from pysam import FastaFile

def iter_peaks_and_labels(fname):
    with open(fname) as fp:
        for line in fp:
            data = line.split()
            yield (data[0], int(data[1]), int(data[2])), data[3]
    return

def main():
    min_region_size = 1000
    genome = FastaFile("GRCh38.genome.fa")
    for region, label in iter_peaks_and_labels(sys.argv[1]):
        # create a new region exactly min_region_size basepairs long centered on 
        # region  
        expanded_start = region[1] + (region[2] - region[1])/2 - min_region_size/2
        expanded_stop = expanded_start + min_region_size
        region = (region[0], expanded_start, expanded_stop)
        print region, label
        print genome.fetch(*region)
    return

if __name__ == '__main__':
    main()
