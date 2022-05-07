import numpy as np
import matplotlib.pyplot as plt

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''
# To compute the centroid of each class (0, 1, 2)
# given data NxM  
def find_centroid(data):
    # sum of the all the data points in the row axis
    # basically find the mean for each class 
    # data.shape[0] #how many rows 75x3 => 1x3
    return np.mean(data, axis = 0, dtype = 'float')

# used the centroid for each class class1 and class 2 (1x3)
# find the distance between pairs
def boundaries(class1, class2):
    boundaries = np.dot((class1 + class2), (class1 - class2))/2
    return boundaries

def prediction(test_data, w, t):
    n = test_data.shape[0] #75x3 = 75
    prediction = np.zeros((n, 1))
    for i in range(n):
        x = test_data[i] #1x3
        if (np.dot(x, w[0])) > t[0]:
            if (np.dot(x, w[1])) > t[1]:
                prediction[i] = 0
            else:
                prediction[i] = 2
        else:
            if (np.dot(x, w[2])) > t[2]:
                prediction[i] = 1
            else:
                prediction[i] = 2

    # save prediction to prediction file
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    return prediction

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):

    # load train data and label
    train_data = np.loadtxt(train_input_dir, skiprows=0)
    train_labels = np.loadtxt(train_label_dir, skiprows=0)

    #find the datapoints for each class
    class0 = train_data[train_labels == 0]
    class1 = train_data[train_labels == 1]
    class2 = train_data[train_labels == 2]

    #print data points that is centroid for each class.
    centroid0 = find_centroid(class0)
    centroid1 = find_centroid(class1)
    centroid2 = find_centroid(class2)
    # print("centroid of class0", centroid0)
    # print("centroid of class1", centroid1)
    # print("centroid of class2", centroid2)

    # find mid point
    m = [0, 0, 0]
    m[0] = (centroid0 + centroid1)/2
    m[1] = (centroid0 + centroid2)/2
    m[2] = (centroid1 + centroid2)/2
    # print("midpoint01", m[0])
    # print("midpoint02", m[1])
    # print("midpoint12", m[2])

    # find w, which is the vector btw one centroid and another
    w = [0, 0, 0]
    w[0] = centroid0 - centroid1
    w[1] = centroid0 - centroid2
    w[2] = centroid1 - centroid2
    # print("w vector for 01", w[0])
    # print("w vector for 02", w[1])
    # print("w vector for 12", w[2])

    # find threshold w*m to compare when prediction
    t = [0, 0, 0]
    t[0] = np.dot(w[0], m[0])
    t[1] = np.dot(w[1], m[1])
    t[2] = np.dot(w[2], m[2])
    # print("threshold for 01:", t[0])
    # print("threshold for 02:", t[1])
    # print("threshold for 12: ", t[2])

    # load test data 
    test_data = np.loadtxt(test_input_dir, skiprows=0)

    # prediction
    prediction(test_data, w, t)
    
if __name__ == "__main__":
    train_input_dir = "../reference/data/training1.txt"
    train_label_dir = "../reference/data/training1_label.txt"
    test_input_dir = "../reference/data/testing1.txt"
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)

