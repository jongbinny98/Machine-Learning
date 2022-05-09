import pandas as pd
import numpy as np
import argparse

from collections import Counter


#TODO: In the below function, write to a file and save so it doesnt need to be done everytime
def preProcess(path):
    """Tokenize each sentence and remove tokens that occur less than 3 times
        
        Arguments:
            path String -- path of the file containg the data we would like to tokenize
        
        Returns:
            array -- an array, where each index is a tokenized, preprocessed sentence
    """
    file = open(path, "r")

    # i = 0
    lines = []
    wordCount = Counter()
    for line in file.readlines():
        # if i < 1:
        # tokenize each sentence
        line = line.strip().split()
        lines.append(line)

        #update counts of each token
        for token in line:
                wordCount[token] += 1
        
        wordCount["<STOP>"] += 1
            # i += 1
        
    #loop through each sentence
    for line in lines:
        #loop through each token in the sentence
        for j in range(0, len(line)):
            if wordCount[line[j]] < 3:
                #remove this word from the count
                del wordCount[line[j]]
                #update it in our data
                line[j] = "<UNK>"
                #increase the count
                wordCount["<UNK>"] += 1

    
    file.close()
    # print(wordCount)
    print(len(wordCount))
    return lines






def main():

    probabilites = {}
    train_tokens = preProcess("./A2-Data/1b_benchmark.train.tokens")
    

    


if __name__ == '__main__':
    main()