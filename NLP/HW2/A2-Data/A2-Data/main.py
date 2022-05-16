import pandas as pd
import numpy as np
import argparse
import random
from collections import Counter
from utils import *

def files_to_text(path):
    train_file = open(path + '1b_benchmark.train.tokens', 'r', encoding='UTF-8')
    dev_file = open(path + '1b_benchmark.dev.tokens', 'r', encoding='UTF-8')
    test_file = open(path + '1b_benchmark.test.tokens', 'r', encoding='UTF-8')
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


def test(M, text):
    p1 = M.ngramPerp(text, 1)
    print("   Unigram Perplexity:", p1)
    p2 = M.ngramPerp(text, 2)
    print("   Bigram Perplexity:", p2)
    p3 = M.ngramPerp(text, 3)
    print("   Trigram Perplexity:", p3)


def testSm(M, text, sm=(1, 1, 1)):
    p1 = M.ngramSmPerp(text, 1, sm[0])
    print("   sm+" + str(sm[0]) + " Unigram Perplexity:", p1)
    p2 = M.ngramSmPerp(text, 2, sm[1])
    print("   sm+" + str(sm[1]) + " Bigram Perplexity:", p2)
    p3 = M.ngramSmPerp(text, 3, sm[2])
    print("   sm+" + str(sm[2]) + " Trigram Perplexity:", p3)


def linInterpOpt(M, text, cycles=5):
    bestLi = (None, float('inf'))
    random.seed(5)
    for i in range(cycles):
        a = random.random()
        b = (1 - a) * random.random()
        c = 1 - a - b
        print("   testing permutations of (%2f, %2f, %2f)" % (a, b, c))
        for li in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
            p = M.linInterpPerp(text, li)
            if p < bestLi[1]:
                bestLi = (li, p)
    return bestLi


def smoothingOpt(M, text, maxSm=(1, 1, 1), dr=(8, 4, 2), cycles=16):
    bestSm = [(None, float('inf')), (None, float('inf')), (None, float('inf'))]
    for i in range(cycles):
        print("   testing sm+(%2f, %2f, %2f)" % (*maxSm,))
        for n in range(1, 4):
            p = M.ngramSmPerp(text, n, maxSm[n-1])
            if p < bestSm[n-1][1]:
                bestSm[n-1] = (maxSm[n-1], p)
        maxSm = (maxSm[0]/dr[0], maxSm[1]/dr[1], maxSm[2]/dr[2])
    return bestSm


def main():
    train_text, test_text, dev_text = files_to_text('./A2-Data/')
    check_text = [["HDTV", "."]]

    # initialize MLE with train_tokens
    print("\nInitializing MLE with train_tokens")
    M = MLE()
    types = M.fit(train_text)
    print ("Preprocessor made", types, "types from train_text (including <STOP> and <UNK>)")

    # Part 1 tests
    print("\n--== Part 1 tests ==--")

    print("\nTesting MLE on: train_text")
    test(M, train_text)
    print("\nTesting MLE on: dev_text")
    test(M, dev_text)
    print("\nTesting MLE on: test_text")
    test(M, test_text)
    print("\nTesting MLE on:", check_text)
    test(M, check_text)

    # Part 2 tests
    print("\n--== Part 2 tests ==--")

    test_sm = [1, 0.1, 5]
    # test_sm = [1]
    print("\nTesting MLE+smoothing on: train_text")
    for sm in test_sm:
        testSm(M, train_text, (sm, sm, sm))
    print("\nTesting MLE+smoothing on: dev_text")
    for sm in test_sm:
        testSm(M, dev_text, (sm, sm, sm))

    print("\nOptimizing MLE+smoothing with dev_text")
    opSm = smoothingOpt(M, dev_text, maxSm=(1, 0.1, 0.01), dr=(8, 2, 2), cycles=8)
    # opSm = ((0.00000005, 892.2466481736056), (0.003125, 418.45086821444113), (0.00125, 2313.2932543498073))
    print("\nOptimized smoothing results on: dev_text")
    print("   sm+" + str(opSm[0][0]) + " Unigram Perplexity:", str(opSm[0][1]))
    print("   sm+" + str(opSm[1][0]) + " Bigram Perplexity:", str(opSm[1][1]))
    print("   sm+" + str(opSm[2][0]) + " Trigram Perplexity:", str(opSm[2][1]))
    opSm = (opSm[0][0], opSm[1][0], opSm[2][0])

    print("\nTesting MLE+smoothing on: test_text")
    testSm(M, test_text, opSm)

    # Part 3 tests
    print("\n--== Part 3 tests ==--")

    # delivery 1
    test_li = [(.7, .1, .2), (.3, .4, .3), (.1, .4, .5), (.1, .2, .7), (.1, .3, .6)]
    # test_li = [(.1, .3, .6)]
    print("\nTesting MLE+linear interpolation on: train_text")
    for li in test_li:
        print("   li=" + str(li) + " Perplexity:", M.linInterpPerp(train_text, li))
    print("\nTesting MLE+linear interpolation on: dev_text")
    for li in test_li:
        print("   li=" + str(li) + " Perplexity:", M.linInterpPerp(dev_text, li))

    print("\nTesting MLE+linear interpolation on:", check_text)
    print("   li=(0.1, 0.3, 0.6) Perplexity:", M.linInterpPerp(check_text, (.1, .3, .6)))

    # delivery 2
    print("\nOptimizing MLE+linear interpolation with dev_text")
    opLi = linInterpOpt(M, dev_text, cycles=5)
    # opLi = ((0.019883, 0.036760, 0.943357), 15.292531)
    print("\nOptimized linear interpolation results on: dev_text")
    print("   li=" + str(opLi[0]) + " Perplexity:", opLi[1])
    opLi = opLi[0]

    print("\nTesting MLE+linear interpolation on: test_text")
    print("   li=" + str(opLi) + " Perplexity:", M.linInterpPerp(test_text, opLi))

    # delivery 3
    new_train_text = []
    for i in range(len(train_text) // 2):
        new_train_text.append(train_text[i])
    print("\nInitializing MLE with half of train_tokens")
    M.fit(new_train_text)
    print("\nTesting MLE+linear interpolation on: test_text")
    print("   li=" + str(opLi) + " Perplexity:", M.linInterpPerp(test_text, opLi))


    # delivery 4
    print("\nInitializing MLE with unkNum=5")
    M.fit(train_text, unkNum=5)
    print("\nTesting MLE+linear interpolation on: test_text")
    print("   li=" + str(opLi) + " Perplexity:", M.linInterpPerp(test_text, opLi))

    print("\nInitializing MLE with unkNum=2")
    M.fit(train_text, unkNum=2)
    print("\nTesting MLE+linear interpolation on: test_text")
    print("   li=" + str(opLi) + " Perplexity:", M.linInterpPerp(test_text, opLi))





if __name__ == '__main__':
    main()
