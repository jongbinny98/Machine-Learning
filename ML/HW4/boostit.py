import numpy as np
import math


class LinearClassifier:

    def __init__(self):
        
        self.w = None          # orthogonal vector to threshold 


    def train(self, X, y, w):
        # class for each class -1 and 1
        class0 = X[y == -1]
        class1 = X[y == 1]

        # centroid for each class
        centroid0 = self.__compute_centroid(class0, w)
        centroid1 = self.__compute_centroid(class1, w)

        # find weight vector 
        self.w = [0, 0]
        self.w[0] = centroid0 - centroid1
        
        # find mid point 
        self.m = [0, 0]
        self.m[0] = (centroid0 + centroid1)/2

        # find threshold w*m to compare
        self.t = [0, 0]
        self.t[0] = np.dot(self.w[0], self.m[0])

        return self
        

    def predict(self, X, w, t):
        
        n = X.shape[0]
        pred = np.zeros((n, 1))
        
        for i in range(n):
            if np.dot(X[i], w[0]) > t[0]:
                pred[i] = -1
            else:
                pred[i] = 1

        return pred

    def error(self, X, y, weights):

        pred = self.predict(X)
        
        TP = 0
        FN = 0
        TN = 0
        FP = 0

        for i in range(len(y)):
            pred_label = pred[i]
            gt_label = y[i]

            if int(pred_label) == -1:
                if pred_label == gt_label:
                    TN += 1 * weights[i]
                else:
                    FN += 1 * weights[i]
            else:
                if pred_label == gt_label:
                    TP += 1 * weights[i]
                else:
                    FP += 1 * weights[i]

        err = 1 - ( (TP + TN) / (TP + FN + FP + TN) )

        return err, pred

    def __compute_centroid(self, w, X):

        c = np.sum(w * X, axis=0) / np.sum(w, axis=0)
        return c


# boosting algorithm
class BoostingClassifier:
    # initialize the parameters here
    def __init__(self):

        self.w = None
        # esemble size
        self.T = 5
        # learning algorithm
        self.A = LinearClassifier()

        # create an empty list of models
        self.M = np.zeros(self.T)
        self.alpha = np.zeros(self.T)

    def fit(self, X, y):

        w = np.empty(X.shape[0])

        # initialize weights
        w[1] = 1 / X.shape[0]
        
        for i in range(1, self.T):
            if True:
                print("Iteration " + str(i) + ":")
            self.M[i] = self.A.train(X, y, w[i])
            err, pred = self.M[i].error(X, y, w[i])

            if True:
                print("Error = " + str(err))

        return self

    def predict(self, X):
        
        return np.ones(X.shape[0], dtype=int)
    
