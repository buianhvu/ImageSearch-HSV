import numpy as np
import csv

class Searcher:
    def __init__(self, indexPath):
        #link to the csv file
        self.indexPath = indexPath
    def search(self, queryFeatures, limit):
        result = {}
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)
                result[row[0]] = d
            f.close
        result = sorted([(v, k) for (k, v) in result.items()])
        return result[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d
