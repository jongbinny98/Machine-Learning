# import the required packages here 
import numpy as np
from math import sqrt

# Calculate the Euclidean distance between two vectors
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum(x1 - x2)**2)  

def find_neighbors(x_train, test, k):


  

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file): 
  '''The function to run your ML algorithm on given datasets, generate 
  the predictions and save them into the provided file path 

  Parameters 
  ---------- 
  Xtrain_file: string 
     Each row is a feature vector. The values in the -th columns are float numbers in the -th dimension
  Ytrain_file: string 
    The CSV file provides the multi-class labels for corresponding feature vectors in the file Xtrain.csv .
    Please note the labels will be integer numbers between 0 and 10. 
  test_data_file: string 
    the path to test data csv file 
  pred_file: string 
    the prediction file name to be saved by your code.
  Ytrain_file 
  ''' 
  # read data from Xtrain_file, Ytrain_file.
  # read data from Xtrain_file, Ytrain_file.
  x = np.loadtxt(Xtrain_file, delimiter=',')
  y = np.loadtxt(Ytrain_file, delimiter=',')

  train_test_split = int(0.9 * len(x))
  x_train, y_label = x[:train_test_split], y[:train_test_split]
  test, test_label = x[train_test_split:], y[train_test_split:]
  
  test =  test[0]
  for i in range(len(x_train)):
    distance = euclidean_distance(x_train[i], test)
    print(distance)
  
  # find the closest neighbors 3
  k = 3
  
  # find the neighbor 
  neighbors = find_neighbors(x_train, test, k)
  for n in range(len(neighbors)):
    print(neighbors[n])


  # save pred_file
  # np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

# define other functions here
if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = None
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)