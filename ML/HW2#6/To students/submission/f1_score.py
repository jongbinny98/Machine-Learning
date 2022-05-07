import numpy as np
def main(predict_label, truth):

    cal = np.zeros((3, 3), dtype = int)

    #open the label file
    true_label = open(truth).readlines()
    predict_label = open(predict_label).readlines()

    # print in 3x3 matrix tp for 00, 11, 22
    for i in range (np.size(true_label)):
        pred = int(predict_label[i])
        act = int(true_label[i])
        cal[pred][act] += 1
    print(cal)
    
    # precision TP / P(predicted)
    Ph_0 = cal[0][0] + cal[0][1] + cal[0][2]
    Ph_1 = cal[1][0] + cal[1][1] + cal[1][2]
    Ph_2 = cal[2][0] + cal[2][1] + cal[2][2]
    TP_0 = cal[0][0]
    TP_1 = cal[1][1]
    TP_2 = cal[2][2]
    precision = ((TP_0/Ph_0)+(TP_1/Ph_1)+(TP_2/Ph_2))/3
    print("precision: ", precision)

    # recall = TPR TP / P(truth)
    P_0 = cal[0][0] + cal[1][0] + cal[2][0]
    P_1 = cal[0][1] + cal[1][1] + cal[2][1]
    P_2 = cal[0][2] + cal[1][2] + cal[2][2]
    recall = ((TP_0/P_0)+(TP_1/P_1)+(TP_2/P_2))/3
    print("recall: ", recall)
    
    #F1_score = 2*precision*recall / precision+recall 2*TP/p(truth)+p(predicted)
    F1_score = (2*precision*recall) / (precision + recall)
    print("F1 score: ", F1_score)

    accuracy = (TP_0 + TP_1 + TP_2) / np.size(true_label)
    print("accuracy: ", accuracy)

if __name__ == '__main__':
    truth  = '../reference/data/testing1_label.txt'
    predict_label  = 'result'
    main(truth, predict_label )
