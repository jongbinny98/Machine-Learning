import numpy as np
import math

class LinearClassifier:

    def __init__(self):
    
        self.t = []  # threshold for class prediction
        self.w = []          # orthogonal vector to threshold 

    def train(self, X, y, weights):

        n_features = X.shape[1]
        n_examples = X.shape[0]
    
        x = np.zeros(shape = (n_examples, n_features))

        # X * weights
        for i in range(n_examples):
            x[i] = X[i] * weights[i]
        
        # class -1
        class0 = x[y == -1]
        class0_weights = weights[y == -1]

        # class 1
        class1 = x[y == 1]
        class1_weights = weights[y == 1]

        # find the centroid
        centroid0 = self.class_exemplar(class0_weights, class0)
        centroid1 = self.class_exemplar(class1_weights, class1)

        # find mid point
        m = []
        m = (centroid0 + centroid1)/2
        print("mid: ", m)

        # find the weight vector
        self.w = []
        self.w = centroid0 - centroid1
        print("w vector", self.w)

        # find threshold
        self.t = []
        self.t = np.dot(self.w, m)
        print("threshold:", self.t)

        return self

    def predict(self, X):
        # print(X)
        n_samples = X.shape[0]
        pred = np.zeros((n_samples, 1))
        # print("X[0]",X[0])
        # print("Xw", np.dot(X[0], self.w))

        for i in range(n_samples):    
            if np.dot(X[i], self.w) > self.t:
                pred[i] = -1
            else:
                pred[i] = 1
        # print(pred)
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

        accuracy = (TP + TN) / (TP + FN + FP + TN)
        precision = TP / (TP + FP) if ((TP + FP) > 0) else 0
        recall = TP / (TP + FN) if ((TP + FN)) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if ((precision + recall) > 0) else 0
        final_score = 50 * accuracy + 50 * f1
        err = 1 - ( (TP + TN) / (TP + FN + FP + TN) )
        print("accuracy: ",accuracy)
        print("precision: ", precision)
        print("final score: ", final_score)
        return err, pred

    def class_exemplar(self, w, x):
        centroid = np.sum(x, axis = 0) / np.sum(w)
        return centroid

class BoostingClassifier:

    def __init__(self):
        
        self.T = 5
        self.A = LinearClassifier

        # create an empty list of models
        self.M = [None for i in range(self.T + 1)]
        self.alpha = [None for i in range(self.T + 1)]

    def fit(self, X, y):

        n_examples = X.shape[0]   # number of samples in training set

        w = np.zeros(shape=(self.T+1, n_examples))
        # print(w.shape)
        # print(w[1])
        # print("-----------------------------------------------")

        w[1] = np.array([1 / n_examples for i in range(n_examples)])  # initialize weights 
        # print(w[1])
        # iterate over T training instances
        for t in range(1, self.T + 1):

            if True:
                print("Iteration " + str(t) + ":")

            # find the model that has trained 
            self.M[t] = self.A().train(X, y, w[t])  
            # find the error rate
            error, pred = self.M[t].error(X, y, w[t])    
            
            if True:
                print("Error = " + str(error))
            
            # if err is more than half, then it will harm it so break
            if error >= 1/2:                   
                self.T = t - 1
                break;

            # compute the alpha
            self.alpha[t] = (1/2) * math.log((1 - error) / error)  
            
            if True:
                print("Alpha = " + str(self.alpha[t]))

                f_inc = 1 / (2 * error)
                f_dec = 1 / (2 * (1 - error))

                print("Factor to increase weights = " + str(f_inc))
                print("Factor to decrease weights = " + str(f_dec))

            # increase or decrease the weights
            if t != self.T:
                for i in range(n_examples):
                    # for misclassified instances
                    if y[i] * pred[i] < 0:  
                        w[t+1][i] = w[t][i] / (2 * error)
                    # for correctly classified instances
                    else:                               
                       w[t+1][i] = w[t][i] / (2 * (1 - error))

        return self

    # prediction
    def predict(self, X):

        prediction = np.zeros(shape=(X.shape[0], 1))

        for i in range(1, self.T+1):
            model = np.sign(self.alpha[i] * self.M[i].predict(X))
            prediction += model
        
        return np.sign(prediction)




