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
    def fit(self, text_set, unkNum=0):
        # count all the words
        wordCount = Counter()
        for line in text_set:
            for word in line:
                wordCount[word] += 1

        # create tokenMap
        self.__init__() # reset tokenMap and tokenList
        index = 2
        for word in wordCount:
            if (wordCount[word] >= unkNum):
                self.tokenMap[word] = index
                self.tokenList.append(word)
                index += 1
        self.tokenList.append('<START>') # to be accessed by tokenList[-1]
        return index

    # tokenizes the text
    def tokenize(self, text_set):
        lines = []
        for sentence in text_set:
            tokens = []
            for word in sentence:
                if not word in self.tokenMap:
                    word = '<UNK>'
                tokens.append(self.tokenMap[word])
            tokens.append(self.tokenMap['<STOP>'])
            lines.append(tokens)
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


class MLE(object):
    def __init__(self):
        self.startToken = -1
        self.pp = PreProcessor()


    def getNgram(self, sentence, index, n):
        ngram = tuple()
        for i in range(max(-1, index - n + 1), index + 1):
            if i == -1:
                ngram += (self.startToken,)
            else:
                ngram += (sentence[i],)
        return ngram


    def pNgram(self, ngram):
        return ngram[:len(ngram) - 1]


    def fit(self, text_set, maxN=3, unkNum=3):
        self.maxN = maxN
        self.v = self.pp.fit(text_set, unkNum) # vocab size
        self.ngramCt = Counter() # number of times each ngram occurs in the text_set
        self.numTokens = 0 # number of tokens in the text_set
        tokens = self.pp.tokenize(text_set)
        for sentence in tokens:
            self.numTokens += len(sentence)
            for n in range(1, self.maxN + 1):
                for i in range(n - 2, len(sentence)):
                    self.ngramCt[self.getNgram(sentence, i, n)] += 1
        return self.v


    def ngramProb(self, ngram):
        top = self.ngramCt[ngram]
        if len(ngram) == 1:
            bottom = self.numTokens
        else:
            bottom = self.ngramCt[self.pNgram(ngram)]
        return top, bottom


    def ngramPerp(self, text_set, n):
        tokens = self.pp.tokenize(text_set)
        logSum = 0
        length = 0
        for sentence in tokens:
            for i in range(len(sentence)):
                ngram = self.getNgram(sentence, i, n)
                top, bottom = self.ngramProb(ngram)
                if top > 0:
                    logSum += np.log(top / bottom)
                    length += 1
        return np.exp(-logSum / length)


    def ngramSmPerp(self, text_set, n, sm=1):
        assert sm > 0

        tokens = self.pp.tokenize(text_set)
        logSum = 0
        length = 0
        for sentence in tokens:
            for i in range(len(sentence)):
                ngram = self.getNgram(sentence, i, n)
                top, bottom = self.ngramProb(ngram)
                logSum += np.log((top + sm) / (bottom + sm * self.v))
            length += len(sentence)
        return np.exp(-logSum / length)


    def linInterpPerp(self, text_set, li=None):
        if li == None:
            li = [1 / self.maxN] * self.maxN
        assert np.abs(1 - np.sum(li)) < 0.00001
        assert len(li) == self.maxN

        tokens = self.pp.tokenize(text_set)
        logSum = 0
        length = 0
        for sentence in tokens:
            for i in range(len(sentence)):
                ngram = self.getNgram(sentence, i, self.maxN)
                top, bottom = self.ngramProb(ngram)
                if top > 0: # if largest ngram exists in train data, all ngrams exist in train data
                    prob = li[self.maxN - 1] * top / bottom
                    for n in range(1, self.maxN):
                        ngram = self.getNgram(sentence, i, n)
                        top, bottom = self.ngramProb(ngram)
                        prob += li[n - 1] * top / bottom
                    logSum += np.log(prob)
                    length += 1
        return np.exp(-logSum / length)


    def linInterpSmPerp(self, text_set, li=None, sm=None):
        if li == None:
            li = [1 / self.maxN] * self.maxN
        assert np.abs(1 - np.sum(li)) < 0.00001
        assert len(li) == self.maxN
        if sm == None:
            sm = [1] * self.maxN
        for a in sm:
            assert a > 0
        assert len(sm) == self.maxN

        tokens = self.pp.tokenize(text_set)
        logSum = 0
        length = 0
        for sentence in tokens:
            for i in range(len(sentence)):
                prob = 0
                for n in range(1, self.maxN + 1):
                    ngram = self.getNgram(sentence, i, n)
                    top, bottom = self.ngramProb(ngram)
                    prob += li[n-1] * (top + sm[n-1]) / (bottom + sm[n-1] * self.v)
                logSum += np.log(prob)
            length += len(sentence)
        return np.exp(-logSum / length)






        
