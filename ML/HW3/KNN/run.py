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
  x_train = np.loadtxt(Xtrain_file, delimiter=',')
  y_label = np.loadtxt(Ytrain_file, delimiter=',')

  # read data from test_data_file(Xtrain.csv)
  test = np.loadtxt(test_data_file, delimiter=',')

  # set k-th closest
  k = 2

  # array of neighbors labels(rough before applying majority rule)
  neighbors = []
  for i in range(len(test)):
    test_value = test[i, :]
    labels = find_neighbors(x_train, y_label, test_value, k)
    neighbors.append(labels)

  prediction = np.zeros(len(test))
  temp = np.zeros(0)

  for i in range(len(test)):
    label, count = np.unique(neighbors[i], return_counts = True)
    temp = label[np.argmax(count)]
    prediction[i] = temp

  #test
  print("....", neighbors[1])
  print("prediction: ", prediction)
  print("y_label: ", y_label)

  print("prediction =", prediction == y_label)

  # save pred_file
  np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = "Xtrain.csv"
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)