
import numpy as np
# class Voted_perceptron:
    
#     def __init__(self):
#         self.t = 0
#         self.T = 5
#         self.k = 0
#         self.c = [0]

#     # find the error rate
#     def error(self, X, y, weights):
        
#         pred = self.predict(X, y ,weights)
        
#         TP = 0
#         FN = 0
#         TN = 0
#         FP = 0

#         # confusion matrix form local_evaluation.py
#         for i in range(len(y)):
#             pred_label = pred[i]
#             gt_label = y[i]

#             if int(pred_label) == -1:
#                 if pred_label == gt_label:
#                     TN += 1 * weights[i]
#                 else:
#                     FN += 1 * weights[i]
#             else:
#                 if pred_label == gt_label:
#                     TP += 1 * weights[i]
#                 else:
#                     FP += 1 * weights[i]

#         # calculation
#         accuracy = (TP + TN) / (TP + FN + FP + TN)
#         precision = TP / (TP + FP) if ((TP + FP) > 0) else 0
#         recall = TP / (TP + FN) if ((TP + FN)) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if ((precision + recall) > 0) else 0
#         final_score = 50 * accuracy + 50 * f1

#         # error rate
#         err = 1 - ( (TP + TN) / (TP + FN + FP + TN) )

#         print("accuracy: ", accuracy)
#         print("precision: ", precision)
#         print("final score: ", final_score)
#         print("--------------------------------------")

#         return err, pred

#     # train
#     def train(self, X, y, w):

#         w = [np.zeros(X.shape[1])]

#         # voted perceptron
#         while self.t < self.T:
#             # for each training example
#             for i in range(len(X)):
#                 inner = np.dot(w[self.k], X[i])
#                 pred = np.sign(X[i] * inner)
#             # misclassification
#             if pred <= 0:
#                 w.append(w[self.k] + np.dot(y[i], X[i]))
#                 self.c.append(1)
#                 self.k += 1
#             else:
#                 self.c[self.k] += 1
#         self.t += 1

#         return self

#     def predict(self, X, y, w):

#         # voted perceptron
#         while self.t < self.T:
#             # for each training example
#             for i in range(len(X)):
#                 inner = np.dot(w[self.k], X[i])
#                 pred = np.sign(y[i] * inner)
#                 # misclassification
#                 if pred <= 0:
#                     w.append(w[self.k] + np.dot(y[i], X[i]))
#                     self.c.append(1)
#                     self.k += 1
#                 else:
#                     self.c[self.k] += 1
#             self.t += 1

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

        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("final score: ", final_score)
        print("--------------------------------------")

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

    # prediction
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

'''
boosting Pseudo code 5-24.17 - train an ensemble of binary classifier from re-weighted training set 

input - data set X, ensemble size T, learning algorithm A(linear classifier)
output - weighted ensemble of models
'''
class BoostingClassifier:

    def __init__(self):

        # Ensemble size
        self.T = 5
        # learning algorithm(Linear Classifier)
        self.A = LinearClassifier
        # model that need to be output 
        self.M = [None for i in range(self.T + 1)]
        # confidence for this model
        self.alpha = [None for i in range(self.T + 1)]

    # boosting algorithm to train
    def fit(self, X, y):
       
        n_examples = X.shape[0]
        # initialize weights (Ensemble size + 1, n_example) ----> (6, 80)
        w = np.zeros((self.T + 1, n_examples))
        
        # start from uniform weight w1i = 1/dataset
        w[1] = np.array([1 / n_examples for i in range(n_examples)])
        
        # run algorithm on data X with weight wti to produce a model Mt
        for t in range(1, self.T + 1):
            
            # print num of current iteration
            print(f"\nIteration {t}:")

            # find the model that has trained 
            self.M[t] = self.A().train(X, y, w[t])  
            # find the error rate
            error, pred = self.M[t].error(X, y, w[t])    
            
            # print error 
            print(f"Error = {error}")
            
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




