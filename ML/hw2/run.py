import numpy as np

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):
    # Reading data
    test_data = np.loadtxt(test_input_dir,skiprows=0)

    [num, _] = test_data.shape

    prediction = np.zeros((num, 1), dtype=np.int16)

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    train_input_dir = 'training1.txt'
    train_label_dir = 'training1_label.txt'
    test_input_dir = 'testing1.txt'
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)
