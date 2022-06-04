
import numpy as np
import random

'''
This is basic linear classifier from hw 2
'''
class LinearClassifier:

    def __init__(self):
        # initialize threshold and w vector
        self.t = []  
        self.w = [] 

    # find the centroid for each class -1,1
    def class_exemplar(self, w, x):
        # instead of using mean(), use the equation 1/sum(w) * sum(w*x)
        centroid = np.sum(x, axis = 0) / np.sum(w)
        # centroid - 0.5 ---->95% final --->51.25% colab
        return centroid

    # find the error rate
    def error(self, X, y, weights):
        
        pred = self.predict(X)
        
        TP = 0
        FN = 0
        TN = 0
        FP = 0

        # confusion matrix form local_evaluation.py
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

        # calculation
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        precision = TP / (TP + FP) if ((TP + FP) > 0) else 0
        recall = TP / (TP + FN) if ((TP + FN)) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if ((precision + recall) > 0) else 0
        final_score = 50 * accuracy + 50 * f1

        # error rate
        err = 1 - ( (TP + TN) / (TP + FN + FP + TN) )

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall", recall)
        print("Final score:", final_score)
        print("Error rate", err)

        return err, pred

    # train
    def train(self, X, y, weights):
        
        n_features = X.shape[1]
        n_examples = X.shape[0]
    
        x = np.zeros((n_examples, n_features))
            
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
        # print("mid: ", m)

        # find the weight vector
        self.w = []
        self.w = centroid0 - centroid1
        # print("w vector", self.w)

        # find threshold
        self.t = []
        self.t = np.dot(self.w, m)
        # print("threshold:", self.t)

        return self

    # prediction
    def predict(self, X):

        n_samples = X.shape[0]
        pred = np.zeros((n_samples, 1))

        for i in range(n_samples):    
            if np.dot(X[i], self.w) > self.t:
                pred[i] = -1
            else:
                pred[i] = 1
        # print(pred)
        return pred

'''
boosting Pseudo code 5-24.17 - train an ensemble of binary classifier from re-weighted training set 

input - data set X, ensemble size T, learning algorithm A(linear classifier)
output - weighted ensemble of models
'''
class BoostingClassifier:

    def __init__(self):

        # Ensemble size
        self.T = 7
        # learning algorithm(Linear Classifier)
        self.A = LinearClassifier
        # model that need to be output 
        self.M = [None] * (self.T + 1)
        # confidence for this model
        self.alpha = [None] * (self.T + 1)

    # boosting algorithm to train
    def fit(self, X, y):
       
        n_examples = X.shape[0]
        # initialize weights (Ensemble size + 1, n_example) ----> (6, 80)
        w = np.zeros((self.T + 1, n_examples))
        
        # start from uniform weight w1i = 1/dataset
        w[1] = np.full(shape = n_examples, fill_value = 1 / n_examples, dtype = np.float)
        #11,69
        random.seed(99)
        random.shuffle(y)
        random.seed(99)
        random.shuffle(X)

        split = int(0.1 * len(y))
        y1 = y[:split]
        y2 = y[split:]
        Y = np.concatenate((y1, y2), axis = None)

        for i in range(len(y1)):
            if y1[i] == 1:
                y1[i] = -1
            else:
                y1[i] = 1

        print("compare:", Y == y)
        
        # run algorithm on data X with weight wti to produce a model Mt
        for t in range(1, self.T + 1):
            
            # print num of current iteration
            print(f"\nIteration {t}:")

            # tried to make noise 
            if (t == 1):
                self.M[t] = self.A().train(X, Y, w[t])
            else:
                # find the model that has trained 
                self.M[t] = self.A().train(X, y, w[t])

            # find the error rate
            error, pred = self.M[t].error(X, y, w[t])    
            
            # if err is more than half, then it will harm it so break
            if error >= 1/2:                   
                self.T = t - 1
                break;

            # compute the confidence of the model
            self.alpha[t] = (1 / 2) * np.log((1 - error) / error)  
            
            # print confidence of the model 
            print(f"Alpha = {self.alpha[t]}")

            # increase or decrease the weights
            if t < self.T:
                for i in range(n_examples):
                    # for misclassified instances
                    if (y[i] * pred[i]) < 0:  
                        # increase weights for all xi in D
                        w[t + 1][i] = w[t][i] / (2 * error)
                    # for correctly classified instances
                    else:
                        # decrease weights for all xi in D
                        w[t + 1][i] = w[t][i] / (2 * (1 - error))

            # print factor to increase and decrease
            factor_increase = 1 / (2 * error)
            factor_decrease = 1 / (2 * (1 - error))

            print(f"Factor to increase weights = {factor_increase}")
            print(f"Factor to decrease weights = {factor_decrease}")

        return self

    # prediction
    def predict(self, X):

        prediction = np.zeros((X.shape[0], 1))

        # return sum of (alpha * Mx) ---> sign(output)
        for i in range(1, self.T):
            prediction += self.alpha[i] * self.M[i].predict(X)
           
        return np.sign(prediction)




