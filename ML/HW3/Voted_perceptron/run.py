# import the required packages here 
import numpy as np

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
  x_train, y_label = x[:train_test_split], y[:train_test_split]
  test, test_label = x[train_test_split:], y[train_test_split:]

  # test data file
  test_data_file = test
  
  # current epoch, number of epoch
  t, T = 0, 5 
  # list of weight
  k = 0
  # weight
  c = [0]
  # classification vector 
  w = [np.zeros(x_train.shape[1])]

  for count in range(y_label.shape[0]):
    if y_label[count] == 0:
      y_label[count] = -1

  # voted perceptron
  while t < T:
    # for each training example
    for i in range(len(x_train)):
      inner = np.dot(w[k], x_train[i])
      pred = np.sign(y_label[i] * inner)
      # misclassification
      if pred <= 0:
        w.append(w[k] + np.dot(y_label[i], x_train[i]))
        c.append(1)
        k += 1
      else:
        c[k] += 1
    t += 1

  # print(c)
  # print(w)

  print("test_label: ",test_label)
  # prediction
  prediction = np.zeros(test_data_file.shape[0])
  # print(test_data_file.shape)
  for j in range(len(test_data_file)):
    for K in range(k):
      inner = np.sign(w[K] * test_data_file[j])
    # print("inner: ", inner)
    pred = np.sign(np.sum(c[K] * inner))
    prediction[j] = int(pred)

  # change the all -1 to 0
  for count in range(len(prediction)):
    if prediction[count] == -1:
      prediction[count] = 0

  print(prediction)

  # save prediction to prediction file
  np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

if __name__ == '__main__':
   Xtrain_file = "Xtrain.csv"
   Ytrain_file = "Ytrain.csv"
   test_data_file = None
   pred_file = 'result'
run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
