import numpy as np
import sys
# You need to build your own model here instead of using well-built python packages such as sklearn
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier

    """
    def __init__(self):
        pass

    def fit(self, X, Y):
        dataSize, vocabSize = X.shape

        # smoothing
        sm = 1

        # split up 0 and 1 classifications
        x0 = X[Y == 0]
        x1 = X[Y == 1]

        # total word count in each classification
        words0 = np.sum(np.sum(x0))
        words1 = np.sum(np.sum(x1))

        # prior probablilty
        self.p0 = np.log(len(x0) / dataSize)
        self.p1 = np.log(len(x1) / dataSize)

        # likelihoods of each word
        self.l0 = np.log((np.sum(x0, axis=0) + sm) / (words0 + sm * vocabSize))
        self.l1 = np.log((np.sum(x1, axis=0) + sm) / (words1 + sm * vocabSize))

    def tenWords(self, wordArray):
        # print 10 most and least hatespeech words
        ratio = self.l1 / self.l0
        low10 = np.argpartition(ratio, -10)[-10:]
        high10 = np.argpartition(ratio, 10)[:10]
        print("10 most:")
        for h in high10:
            print(wordArray[h], end=", ")
        print("\n10 least:")
        for l in low10:
            print(wordArray[l], end=", ")
        print()

    def predict(self, X):
        result = []

        for i in range(len(X)):
            sentence = X[i]
            # likelihoods of whole sentence
            likelihood0 = np.sum(np.multiply(sentence, self.l0))
            likelihood1 = np.sum(np.multiply(sentence, self.l1))
            # compare numerator of bayes (since denominators are equal)
            if likelihood0 + self.p0 > likelihood1 + self.p1:
                result.append(0)
            else:
                result.append(1)
        return result


class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        # Add your code here!
        pass
    
    def loadingbar(self, size, current, total):
        percent = current / total
        bars = round(percent * size)
        sys.stdout.write('\r')
        sys.stdout.write('\u2588' * bars + '\u2591' * (size-bars))
        sys.stdout.write(' loading... {0:1.1f}%'.format(percent * 100))
        sys.stdout.flush()
    
    # Run gradient descent for n epochs (n = self.num_epochs),
    # updating theta during each epoch
    def fit(self, X, Y):
        
        self.lambd = 0.001
        self.learning_rate = 0.1
        self.num_epochs = 5000
        self.num_words = X.shape[1]
        self.theta = np.ones(self.num_words)

    
        for n in range(self.num_epochs):

            # step decay adaptive learning rate
            if (n%100 == 0):
                self.learning_rate *= 0.999
                #print("epoch: ", n)
                #print("learning rate: ", self.learning_rate)
            self.gradient_descent_step(X, Y) # update theta
            self.loadingbar(40, n, self.num_epochs)
        print('\n')

    # Predicts the label/class of X by outputting either 1 or 0
    def predict(self, X):
        return np.round(self.sigmoid(X @ self.theta))
        
    # Convert raw model output to probabilities.
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # perform a single gradient update on theta
    def gradient_descent_step(self, X, y):
        self.theta = (self.theta - (self.learning_rate * self.cross_entropy_loss_derivative(X, y)))
    # Calculate the derivative of the loss function with respect to theta.
    # The derivative of the loss function also adds the derivative of the L2 regularization term.
    def cross_entropy_loss_derivative(self, X, y):
        m = X.shape[0]
        reg_term = self.lambd * 2 * np.sum(self.theta[1:]) # compute the L2 regularization term
        return (-1/m) * (y - self.sigmoid(X @ self.theta)) @ X + reg_term


# You change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
