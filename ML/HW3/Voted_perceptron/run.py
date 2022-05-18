# import the required packages here 
import numpy as np
# import matplotlib.pyplot as plt

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

  #one two five ten twenty

  #remaining 99 percent
  one =  int(0.01*len(x_train))
  train_one, label_one = x_train[:one], y_label[:one]
  #remaining 98 percent
  two =  int(0.02*len(x_train))
  train_two, label_two = x_train[:two], y_label[:two]
  #remaining 95 percent
  five =  int(0.05*len(x_train))
  train_five, label_five = x_train[:five], y_label[:five]
  #remaining 90 percent
  ten =  int(0.10*len(x_train))
  train_ten, label_ten = x_train[:ten], y_label[:ten]
  #remaining 80 percent
  twenty =  int(0.20*len(x_train))
  train_twenty, label_twenty = x_train[:twenty], y_label[:twenty]

  # test data file
  test_data_file = test
  
  # current epoch, number of epoch
  t, T = 0, 5 
  # list of weight
  k = 0
  # weight
  c = [0]
  # classification vector 
  w = [np.zeros(train_five.shape[1])]

  for count in range(label_five.shape[0]):
    if label_five[count] == 0:
      label_five[count] = -1

  # voted perceptron
  while t < T:
    # for each training example
    for i in range(len(train_five)):
      inner = np.dot(w[k], train_five[i])
      pred = np.sign(label_five[i] * inner)
      # misclassification
      if pred <= 0:
        w.append(w[k] + np.dot(y[i], train_five[i]))
        c.append(1)
        k += 1
      else:
        c[k] += 1
    t += 1
  
  for p in range(label_five.shape[0]):
    if label_five[p] == -1:
      label_five[p] = 0

  prediction = np.zeros(len(test_data_file))
  for j in range(len(test_data_file)):
    for K in range(k):
      inner = np.sign(w[K] * test_data_file[j])
      pred = np.sign(np.sum(c[K] * inner))
    prediction[j] = pred

  # change the all -1 back to 0
  for count in range(prediction.shape[0]):
    if prediction[count] == -1:
      prediction[count] = 0

  print("compare y and pred \n", test_label == prediction)

  # save pred_file
  np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")
  
  acc = np.sum(np.equal(test_label, prediction)) / len(test_label)
  
  print(acc)
#   main(prediction, test_label)

# #prediction and test_label = 10 percent truth label
# def main(prediction, test_label):

#     cal = np.zeros((2, 2), dtype = int)

#     # print in 2x2 matrix tp for 00, 11
#     for i in range (np.size(test_label)):
#         pred = int(prediction[i])
#         true = int(test_label[i])
#         cal[pred][true] += 1
#     print(cal)
    
#     TP_0 = cal[0][0]
#     TP_1 = cal[1][1]

#     accuracy = (TP_0 + TP_1) / np.size(test_label)
#     print("accuracy: ", accuracy)

if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = None
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
