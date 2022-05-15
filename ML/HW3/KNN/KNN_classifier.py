# import the required packages here 
import numpy as np

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file): 
  '''The function to run your ML algorithm on given datasets, generate 
  the predictions and save them into the provided file path 
   
  Parameters 
  ---------- 
  Xtrain_file: string 
    the path to Xtrain csv file 
  Ytrain_file: string 
    the path to Ytrain csv file 
  test_data_file: string 
    the path to test data csv file 
  pred_file: string 
    the prediction file name to be saved by your code. You have to 
  save your predictions into this file path following the same format of 
  Ytrain_file 
  ''' 
  # read data from Xtrain_file, Ytrain_file.
  x = np.loadtxt(Xtrain_file, delimiter=',')
  y = np.loadtxt(Ytrain_file, delimiter=',')
 
 
# define other functions here
if __name__ == '__main__':

  Xtrain_file = "Xtrain.csv"
  Ytrain_file = "Ytrain.csv"
  test_data_file = "Xtrain.csv"
  pred_file = 'result'

  run(Xtrain_file, Ytrain_file, test_data_file, pred_file)