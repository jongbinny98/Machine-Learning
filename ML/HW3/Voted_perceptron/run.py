# import the required packages here 
import numpy as np

# def prediction(x, w, c, k, pred_file):

#   prediction = np.zeros(x.shape[0])
#   for j in range(len(x)):
#     for K in range(k):
#       inner = np.sign(w[K] * x[j])
#       pred = np.sign(np.sum(c[K] * inner))
#     prediction[j] = pred

#   # change the all -1 back to 0
#   for count in range(len(prediction)):
#     if prediction[count] == -1:
#       prediction[count] = 0

#   np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

#   return pred_file

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

  # train_test_split = int(0.9 * len(x))
  # x_train, y_label = x[:train_test_split], y[:train_test_split]
  # test, test_label = x[train_test_split:], y[train_test_split:]

  # # test data file
  test = np.loadtxt(test_data_file, delimiter=',')
  
  # current epoch, number of epoch
  t, T = 0, 5 
  # list of weight
  k = 0
  # weight
  c = [0]
  # classification vector 
  w = [np.zeros(x.shape[1])]

  for count in range(y.shape[0]):
    if y[count] == 0:
      y[count] = -1

  # voted perceptron
  while t < T:
    # for each training example
    for i in range(len(x)):
      inner = np.dot(w[k], x[i])
      pred = np.sign(y[i] * inner)
      # misclassification
      if pred <= 0:
        w.append(w[k] + np.dot(y[i], x[i]))
        c.append(1)
        k += 1
      else:
        c[k] += 1
    t += 1
  
  for p in range(y.shape[0]):
    if y[p] == -1:
      y[p] = 0

  # print(prediction)
  prediction = np.zeros(len(test))
  # print(test.shape[0])
  for j in range(len(test)):
    for K in range(k):
      inner = np.sign(w[K] * test[j])
      pred = np.sign(np.sum(c[K] * inner))
    prediction[j] = pred

  # change the all -1 back to 0
  for count in range(prediction.shape[0]):
    if prediction[count] == -1:
      prediction[count] = 0
  # print(len(prediction))
  # print(len(y))
  print("compare y and pred \n", y == prediction)
  # print("pred: \n", prediction)
  # save prediction to prediction file

  np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = "Xtrain.csv"
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
