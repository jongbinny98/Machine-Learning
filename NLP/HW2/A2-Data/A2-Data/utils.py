from collections import Counter
import numpy as np
import math

class PreProcessor(object):
    def __init__(self):
        self.tokenMap = {'<STOP>' : 0, '<UNK>' : 1}
        self.tokenList = ['<STOP>', '<UNK>']

    # creates the tokenMap
    def fit(self, text_set):
        # count all the words
        wordCount = {}
        for line in text_set:
            for word in line:
                wordCount[word] = wordCount.get(word, 0) + 1

        # create tokenMap
        self.tokenMap = {'<STOP>' : 0, '<UNK>' : 1}
        index = 2
        for word in wordCount:
            if (wordCount[word] >= 3):
                self.tokenMap[word] = index
                self.tokenList.append(word)
                index += 1
        self.tokenList.append('<START>')
        print ("preprocessor created", index, "unique tokens including <STOP> and <UNK>")

    def transform_list(self, text_set):
        counts = Counter()
        llist = []
        for i in range(len(text_set)):
            wlist = []
            for j in range(len(text_set[i])):
                word = text_set[i][j]
                if (word in self.tokenMap):
                    token = self.tokenMap[word]
                else:
                    token = self.tokenMap['<UNK>']
                wlist.append(token)
                counts[token] += 1
            llist.append(wlist)
        return llist, counts


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
        self.ngram = {}
        self.ngramList = []
        self.n = n

    def getNgram(self, sentence, index):
        # get the n words for the ngram
        ngram = tuple()
        for offset in range(1 - self.n, 1):
            i = index + offset
            if (i < 0):
                ngram += (-1,) # <START>
            elif (i >= len(sentence)):
                ngram += (0,) # <STOP>
            else:
                ngram += (sentence[i],)
        
        return ngram

    def fit(self, text_set):
        index = 0
        for sentence in text_set:
            for i in range(len(sentence) + self.n - 1):
                # get the ngram at index i
                t = self.getNgram(sentence, i)
                # put the (ngram, index) into the hash map
                if t not in self.ngram:
                    self.ngram[t] = index
                    self.ngramList.append(t)
                    index += 1
                else:
                    continue

    def transform(self, text):
        feature = Counter()
        for i in range(len(text) + self.n - 1):
            t = self.getNgram(text, i)
            if t in self.ngram:
                feature[self.ngram[t]] += 1
        return feature


    def transform_list(self, text_set):
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        return features

    def getPrevNgram(self, ngramIndex):
        ngram = self.ngramList[ngramIndex]
        return ngram[:len(ngram -1)]

        

class MLE(object):
    def __init__(self,features, nGramTotalCounts, nGramCounts, nGramTotals):
        self.sentenceProbs = {}
        self.features = features
        self.nGramTotalCounts = nGramTotalCounts
        self.nGramCounts = nGramCounts
        self.nGramTotals = nGramTotals

    def unigramPerplexity(self):
        feature = self.features[0]
        nGramTotalCount = self.nGramTotalCounts[0]
        nGramCount = self.nGramCounts[0]
        ngramTotal = self.nGramTotals[0]

        
        total = 0
        for counter in nGramCount: #each sentence is a counter
            # print(counter)
            # print(nGramCount)
            for count in counter:
                total += np.log(nGramTotalCount[count] / ngramTotal)
        l = total / ngramTotal

        return np.exp(-l)
     

    def bigramMLE(self):
        pass

    def trigramMLE(self):
        pass
        