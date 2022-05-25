# import the required packages here 
import numpy as np
from math import sqrt

# Calculate the Euclidean distance between two vectors
def euclidean_distance(x1, x2):
  euclidean_distance = np.sum(pow((x1 - x2), 2))
  return np.sqrt(euclidean_distance)  

# To find the k neighbor's label
def find_neighbors(x_train, y_label, test, k):

  # Predict neighbor = k-th closest neighbors
  Predict_Neighbors = np.zeros(k)

  # temp = store the distance btw x_train and test[0] point
  temp = np.zeros(len(x_train))

  # find the distance and use index to find the k-th closest neighbors
  for i in range(len(x_train)):
    distance = euclidean_distance(x_train[i], test)
    temp[i] = distance
    index = temp.argsort()
  
  # find the k-th closest label
  for j in range(k):
    Predict_Neighbors[j] = y_label[index[j]]

  return Predict_Neighbors

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file): 

  # read data from Xtrain_file, Ytrain_file.
  x = np.loadtxt(Xtrain_file, delimiter=',')
  y = np.loadtxt(Ytrain_file, delimiter=',')

  np.random.seed(0)
  np.random.shuffle(x)
  np.random.seed(0)
  np.random.shuffle(y)
  # # read data from test_data_file(Xtrain.csv)
  # test = np.loadtxt(test_data_file, delimiter=',')
  
  train_test_split = int(0.8 * len(x))
  x_train, y_label = x[:train_test_split], y[:train_test_split]
  test, test_label = x[train_test_split:], y[train_test_split:]
  

  # set k-th closest
  k = 30

  # array of neighbors labels(rough before applying majority rule)
  neighbors = []
  for i in range(len(test)):
    # to get each row line 
    test_value = test[i, :]
    labels = find_neighbors(x_train, y_label, test_value, k)
    neighbors.append(labels)
  
  prediction = np.zeros(len(test))
  temp = np.zeros(0)

  for i in range(len(test)):
    ''' np.unique from numpy.unique
    return_counts: If return_counts is True, also return the number of times each unique item appears in ar.
    values: the sorted unique elements of an array
    counts: the number of times each unique item appears in ar
    ''' 
    values, count = np.unique(neighbors[i], return_counts = True)
    temp = values[np.argmax(count)]
    prediction[i] = temp

  # test
  print("Prediction = truth _______________________________________________________ \n", prediction == test_label)
  print("Prediction _______________________________________________________________ \n", prediction)
  print("Truth ____________________________________________________________________ \n", test_label)
  
  # # compute the accuracy
  accuracy = np.sum(np.equal(test_label, prediction)) / len(test_label)
  print("accuracy _________________________________________________________________ \n", accuracy)

  # save pred_file
  np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = "Xtrain.csv"
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)