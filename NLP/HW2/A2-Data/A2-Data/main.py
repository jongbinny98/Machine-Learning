import pandas as pd
import numpy as np
import argparse
from collections import Counter
from utils import *

def files_to_text():
    train_file = open('1b_benchmark.train.tokens', encoding="utf8")
    dev_file = open('1b_benchmark.dev.tokens', encoding="utf8")
    test_file = open('1b_benchmark.test.tokens', encoding="utf8")

    train_text = []
    dev_text = []
    test_text = []

    for line in train_file:
        train_text.append(line.split())
    for line in dev_file:
        dev_text.append(line.split())
    for line in test_file:
        test_text.append(line.split())

    train_file.close()
    dev_file.close()
    test_file.close()

    return train_text, test_text, dev_text

def test(M, tokens):
    print("   Unigram Perplexity:", M.unigramMLE(tokens))
    print("   Bigram Perplexity:", M.bigramMLE(tokens))
    print("   Trigram Perplexity:", M.trigramMLE(tokens))

def main():
    train_text, test_text, dev_text = files_to_text()
    check_text = [["HDTV", "."]]

    # initialize preprocessor with train_text
    pp = PreProcessor()
    numTypes = pp.fit(train_text)
    print ("Preprocessor created", numTypes, "unique tokens (types) including <STOP> and <UNK>")

    # tokenize text with preprocessor
    train_tokens = pp.process(train_text)
    test_tokens = pp.process(test_text)
    dev_tokens = pp.process(dev_text)
    check_tokens = pp.process(check_text)

    # initialize MLE with train_tokens
    print("\ninitializing MLE")
    M = MLE()
    M.fit(train_tokens)

    # tests
    print("\nTesting MLE on: train_text")
    test(M, train_tokens)
    print("\nTesting MLE on: dev_text")
    test(M, dev_tokens)
    print("\nTesting MLE on:", check_text)
    test(M, check_tokens)

    # now with smoothing
    for i in [1, 0.01, 0.0001]:
        print("\ninitializing MLE with +" + str(i) + " smoothing")
        M.fit(train_tokens, i)
        # tests
        print("\nTesting MLE sm+" + str(i) + " on: train_text")
        test(M, train_tokens)
        print("\nTesting MLE sm+" + str(i) + " on: dev_text")
        test(M, dev_tokens)
if __name__ == '__main__':
    main()
