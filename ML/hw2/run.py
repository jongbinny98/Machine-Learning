import numpy as np
import matplotlib.pyplot as plt

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''
#find the centroid 
def centroid(data):
    return np.sum(data) / data.shape[0]

def compute_threshold (class1, class2):
    return np.dot((class1 - class2), (class1 + class2)) / 2

def compute_orth (class1, class2):

    return class1 - class2

def disctriminant (w, x, t):

    if np.dot(w, x) <= t:
        return 0
    else:
        return 1

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):

    #Reading data
    train_data = np.loadtxt(train_input_dir)
    train_label = np.loadtxt(train_label_dir)
    
    #train_data 300(0)x3(1)
    n = train_data[1] #second row 
    print("shape: ", train_data.shape[0]) #300x3 = 300
    
    test_data = np.loadtxt(test_input_dir,skiprows=0)

    # n = test_data.shape[0]
    [num, _] = test_data.shape

    prediction = np.zeros((num, 1), dtype=np.int16)

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    train_input_dir = "../reference/data/training1.txt"
    train_label_dir = "../reference/data/training1_label.txt"
    test_input_dir = "../reference/data/testing1.txt"
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)

