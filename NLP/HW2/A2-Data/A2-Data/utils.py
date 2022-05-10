from collections import Counter
import numpy as np
import math

class PreProcessor(object):
    def __init__(self):
        # key = word, value = token
        self.tokenMap = {'<STOP>' : 0, '<UNK>' : 1}

        # reverse tokenMap: index = token, value = word (used for printing)
        self.tokenList = ['<STOP>', '<UNK>']

    # creates the tokenMap and returns number of types (unique tokens)
    def fit(self, text_set):
        # count all the words
        wordCount = Counter()
        for line in text_set:
            for word in line:
                wordCount[word] += 1

        # create tokenMap
        self.__init__() # reset tokenMap and tokenList
        index = 2
        for word in wordCount:
            if (wordCount[word] >= 3):
                self.tokenMap[word] = index
                self.tokenList.append(word)
                index += 1
        self.tokenList.append('<START>') # to be accessed by tokenList[-1]
        return index

    # tokenizes the text
    def process(self, text_set):
        lines = []
        for i in range(len(text_set)):
            words = []
            for j in range(len(text_set[i])):
                word = text_set[i][j]
                if word in self.tokenMap:
                    token = self.tokenMap[word]
                else:
                    token = self.tokenMap['<UNK>']
                words.append(token)
            token = self.tokenMap['<STOP>']
            words.append(token)
            lines.append(words)
        return lines

    # returns a string representation of an ngram
    def ngram(self, t):
        res = "("
        for i in range(len(t)):
            if i > 0:
                res += " "
            token = t[i]
            res += self.tokenList[token]
        res += ")"
        return res


class FeatureExtractor(object):
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass


# <START> = -1, <STOP> = 0
class NgramFeature(FeatureExtractor):
    def __init__(self, n):
        self.ngramCt = Counter()
        self.totalCt = 0
        self.n = n
        self.startToken = -1

    # returns a tuple with n or less tokens
    def getNgram(self, sentence, index):
        ngram = tuple()
        for offset in range(1 - self.n, 1):
            i = index + offset
            if i == -1:
                ngram += (self.startToken,)
            elif i >= 0:
                ngram += (sentence[i],)
        return ngram

    def fit(self, text_set):
        for text in text_set:
            for i in range(len(text)):
                self.ngramCt[self.getNgram(text, i)] += 1
                self.totalCt += 1
        return len(self.ngramCt)

    def transform(self, text):
        feature = Counter()
        for i in range(len(text)):
            t = self.getNgram(text, i)
            if t in self.ngramCt:
                feature[t] += 1
        return feature

    def transform_list(self, text_set):
        features = []
        for text in text_set:
            features.append(self.transform(text))
        return features


class MLE(object):
    def __init__(self):
        self.N1 = NgramFeature(1)
        self.N2 = NgramFeature(2)
        self.N3 = NgramFeature(3)

    def pNgram(self, ngram):
        return ngram[:len(ngram) - 1]

    def fit(self, text_set, sm=0):
        print("Fitting:", end="  ")
        self.__init__() # reset N1, N2, N3
        print(self.N1.fit(text_set), "unigram types", end=",  ")
        print(self.N2.fit(text_set), "bigram types", end=",  ")
        print(self.N3.fit(text_set), "trigram types")

        print("Calculating N-gram Probabilities:")
        v = len(self.N1.ngramCt) # vocab size

        self.N1.ngramCt[(-1,)] = self.N1.ngramCt[(0,)] # actual bullshit but ok

        self.p1 = {}
        for n in self.N1.ngramCt:
            self.p1[n] = np.log((self.N1.ngramCt[n] + sm) / (self.N1.totalCt + sm * v))
        self.p2 = {}
        for n in self.N2.ngramCt:
            self.p2[n] = np.log((self.N2.ngramCt[n] + sm) / (self.N1.ngramCt[self.pNgram(n)] + sm * v))
        self.p3 = {}
        for n in self.N3.ngramCt:
            if len(n) == 3:
                self.p3[n] = np.log((self.N3.ngramCt[n] + sm) / (self.N2.ngramCt[self.pNgram(n)] + sm * v))
            else:
                self.p3[n] = self.p2[n]

    def unigramMLE(self, text_set):
        return self.ngramMLE(text_set, self.N1, self.p1)

    def bigramMLE(self, text_set):
        return self.ngramMLE(text_set, self.N2, self.p2)

    def trigramMLE(self, text_set):
        return self.ngramMLE(text_set, self.N3, self.p3)

    def ngramMLE(self, text_set, Feature, probs):
        ngram_set = Feature.transform_list(text_set)
        logSum = 0
        length = 0
        for sentence in ngram_set:
            for n in sentence:
                if n in Feature.ngramCt:
                    logSum += sentence[n] * probs[n]
                    length += sentence[n]
        return np.exp(-logSum / length)

        
