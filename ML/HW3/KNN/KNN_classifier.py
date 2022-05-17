# import the required packages here 
import numpy as np
from math import sqrt

# Calculate the Euclidean distance between two vectors
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum(x1 - x2)**2)  

# To find the k neighbor's label
def find_neighbors(x_train, y_label, test, k):

  # Predict neighbor = k-th closest neighbors
  Predict_Neighbors = np.zeros((k))

  # temp = store the distance btw x_train and test[0] point
  temp = np.zeros((len(x_train)))
  distance = np.zeros((len(x_train)))
  temp = np.zeros((len(x_train)))

  # find the distance and use index to find the k-th closest neighbors
  for i in range(len(x_train)):
    for j in range(len(test)):
      distance[j] = euclidean_distance(x_train[i], test[j])
      temp[i] = distance[j]
  print(temp)

  # get the index of temp 
  index = temp.argsort()
  print(index)
  # find the k-th closest label
  for j in range(k):
    Predict_Neighbors[j] = index[j]
    
  return Predict_Neighbors

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file): 

  # read data from Xtrain_file, Ytrain_file.
  x = np.loadtxt(Xtrain_file, delimiter=',')
  y = np.loadtxt(Ytrain_file, delimiter=',')

  #split the data into train(0.90) and test(0.10)
  train_test_split = int(0.9 * len(x))
  x_train, y_label = x[:train_test_split], y[:train_test_split]
  test, test_label = x[train_test_split:], y[train_test_split:]

  # set k-th closest
  k = 3

  # find the k-th closest neighbors label
  neighbors_label = find_neighbors(x_train, y_label, test, k)

  print("neighbor: ", neighbors_label)
  
  # prediction
  # prediction = np.zeros(len(test))  
  # for i in range(len(test)):
  #   if 
  

  # save pred_file
  # np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

# define other functions here
if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = None
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)