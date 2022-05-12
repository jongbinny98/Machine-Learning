# import the required packages here 
import numpy as np
# import math

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file): 
  '''The function to run your ML algorithm on given datasets, generate the predictions and save them into the provided file path
     Parameters
  ----------
  Xtrain_file: string the path to Xtrain csv file
  Ytrain_file: string the path to Ytrain csv file
  test_data_file: string the path to test data csv file
  pred_file: string
  the prediction file to be saved by your code. 
  You have to save your predictions into this file path following the same format of Ytrain_file
  '''
  # read data from Xtrain_file, Ytrain_file.
  x = np.loadtxt(Xtrain_file, delimiter=',')
  y = np.loadtxt(Ytrain_file, delimiter=',')

  train_test_split = int(0.9 * len(x))
  x_train, x_label = x[:train_test_split], y[:train_test_split]
  test, test_label = x[train_test_split:], y[train_test_split:]

  # test data file
  test_data_file = test
  
  # number of epoch
  t, T = 0, 5 
  # list of weight
  k = 0
  # voted
  c = np.zeros(x_train.shape[0])
  # weight # 450 X 1
  w = np.zeros(x_train.shape[0])

  # print(x_train.shape)
  # print(x_train.shape[0])
  # print(x_label.shape)
  # inner = np.dot(w[0], x_train[0])
  # print(inner.shape)
  for count in range(x_label.shape[0]):
    if x_label[count] == 0:
      x_label[count] = -1

  # voted perceptron
  while t < T:
    # for each training example
    for i in range(len(x_train)):
      inner = np.dot(w[k], x_train[i])
      pred = np.sign(x_label[i] * inner)
      if (pred <= 0).all:
        w[k+1] = np.sum(w[k], np.dot(x_label[i], x_train[i]))
        c[k+1] = c[k] + 1
        k += 1
      else:
        c[k] += 1
    t += 1


  # prediction
  prediction = np.zeros(test.shape[0])
  # print(prediction.shape)
  # for j in range(test.shape[0]):



  # save prediction to prediction file
  # np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

if __name__ == '__main__':
  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = None
  pred_file = None
  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
