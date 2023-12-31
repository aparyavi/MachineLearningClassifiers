"""
In dumbClassifiers.py, we implement the world's simplest classifiers:
  1) Always predict +1
  2) Always predict the most frequent label from the training data
  3) Just use the sign of the first feature to decide on label
"""

from binary import *
from numpy  import *

import util

class AlwaysPredictOne(BinaryClassifier):
    """
    This defines the classifier that always predicts +1.
    """

    def __init__(self, opts):
        """
        do nothing
        """

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictOne"

    def predict(self, X):
        return 1       # return our constant prediction

    def train(self, X, Y):
        """
        do nothing
        """


class AlwaysPredictMostFrequent(BinaryClassifier):
    """
    This defines the classifier that always predicts the
    most frequent label from the training data.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, assume most frequent class is +1
        """
        self.mostFrequentClass = 1

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictMostFrequent(%d)" % self.mostFrequentClass

    def predict(self, X):
        """
        X is an vector and we want to make a single prediction: Just
        return the most frequent class!
        """
        return self.mostFrequentClass

    def train(self, X, Y):
        '''
        just figure out what the most frequent class is and store it in self.mostFrequentClass
        '''
        oneCount = 0
        negCount = 1
        for y in Y:
            if y > 0:
                oneCount += 1
            else:
                negCount += 1
            if oneCount >= negCount:
                self.mostFrequentClass = 1
            else:
                self.mostFrequentClass = -1
            
        
        

class FirstFeatureClassifier(BinaryClassifier):
    """
    This defines the classifier that always predicts on the basis of
    the first feature only.  In particular, we maintain two
    predictors: one for when the first feature is >0, one for when the
    first feature is <= 0.
    """

    def __init__(self, opts):
        """
        if we haven't been trained, always return 1
        """
        self.classForPos = 0    # what class should we return if X[0] >  0
        self.classForNeg = 0    # what class should we return if X[0] <= 0

    def online(self):
        return False

    def __repr__(self):
        return "FirstFeatureClassifier(%d,%d)" % (self.classForPos, self.classForNeg)

    def predict(self, X):
        """
        check the first feature and make a classification decision based on it
        """
        forPos = 0
        forNeg = 0
        if self.classForPos >= 0:
            forPos = 1
        else: 
            forPos = -1
        
        if self.classForNeg >= 0:
            forNeg = 1
        else: 
            forNeg = -1
        
        if X[0] > 0:
            return forPos
        else:
            return forNeg
        

    def train(self, X, Y):
        '''
        just figure out what the most frequent class is for each value of X[:,0] and store it
        '''
        for idx, x in enumerate(X):
            if x[0] > 0:
                self.classForPos += Y[idx]
            else:
                self.classForNeg += Y[idx]