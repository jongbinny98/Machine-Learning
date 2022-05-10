import pandas as pd
import numpy as np
import math
#import argparse
from collections import Counter
from utils import *
      

def main(train_dir,dev_dir,test_dir):
    train_file = open(train_dir, encoding='utf-8')
    dev_file = open(dev_dir, 'r')
    test_file = open(test_dir, 'r')

    # Tokenize train_file
    tokenized_text = []
    for line in train_file:
        tokenized_text.append(line.split())

    # preprocess train_file to create tokenMap
    pp = PreProcessor()
    pp.fit(tokenized_text)

    # replace strings with integers for faster computation
    tokenized_text, counts = pp.transform_list(tokenized_text)
    # for line in tokenized_text:
    #     for word in line:
    #         print(pp.tokenList[word])

    # feat_extractor = NgramFeature(3)
    # feat_extractor.fit(tokenized_text)
    # print(len(feat_extractor.ngram))

    # X_train = feat_extractor.transform_list(tokenized_text)
    # print(X_train)
    features = (NgramFeature(1), NgramFeature(2), NgramFeature(3))
    ngramTotals = tuple() # contains integer for the total number of ngrams
    ngramTotalCounts = tuple() # contains array index = ngramNum, value = num of occurences
    ngramCounts = tuple() # contains array of counters, 1 counter for each sentence, key = ngramNum, value = num of occurences

    for feature in features:
        feature.fit(tokenized_text)
        ngramCount = feature.transform_list(tokenized_text)
        ngramCounts += (ngramCount,)
        total = 0
        ngramTotalCount = np.zeros(len(feature.ngram))
        for count in ngramCount:
            for index in count:
                ngramTotalCount[index] += count[index]
                total += count[index]
        ngramTotalCounts += (ngramTotalCount,)
        ngramTotals += (total,)

    #calculate MLE for each sentence
    mle = MLE(features, ngramTotalCounts, ngramCounts, ngramTotals)
    unigramTotal = mle.unigramPerplexity()

    print("unigram_perplexity: ", unigramTotal)

    #calculate Perplexity
if __name__ == '__main__':
    train_dir = "1b_benchmark.train.tokens"
    dev_dir = "1b_benchmark.dev.tokens"
    test_dir = "1b_benchmark.test.tokens"
    main(train_dir,dev_dir,test_dir)