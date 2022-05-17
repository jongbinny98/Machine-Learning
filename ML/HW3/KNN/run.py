# import the required packages here 
import numpy as np
from math import sqrt

# Calculate the Euclidean distance between two vectors
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum(x1 - x2)**2)  

# To find the k neighbor's label
def find_neighbors(x_train, y_label, test_data_file, k):

  # Predict neighbor = k-th closest neighbors
  Predict_Neighbors = np.zeros(k)

  # temp = store the distance btw x_train and test[0] point
  temp = np.zeros(len(x_train))

  # find the distance and use index to find the k-th closest neighbors
  for i in range(len(x_train)):
    distance = euclidean_distance(x_train[i], test_data_file)
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

  #split the data into train(0.90) and test(0.10)
  train_test_split = int(0.9 * len(x))
  x_train, y_label = x[:train_test_split], y[:train_test_split]
  test, test_label = x[train_test_split:], y[train_test_split:]

  #store test into test_data_file
  test_data_file = test

  # set k-th closest
  k = 3

  # array of neighbors labels(rough before applying majority rule)
  neighbors = []
  for i in range(len(test_data_file)):
    test_value = test_data_file[i, :]
    labels = find_neighbors(x_train, y_label, test_value, k)
    neighbors.append(labels)

  prediction = np.zeros(len(test_data_file))
  temp = np.zeros(0)

  for i in range(len(test_data_file)):
    label, count = np.unique(neighbors[i], return_counts = True)
    temp = label[np.argmax(count)]
    prediction[i] = temp
  print("prediction =", prediction == test_label)

  # save pred_file
  np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

# define other functions here
if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = None
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)