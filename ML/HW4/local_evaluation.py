#!/usr/bin/env python3
import sys
import os
import numpy as np
import time

from boostit import BoostingClassifier

# evaluation on your local machine only
dataset_dir = 'dataset1'
train_set = os.path.join(dataset_dir, 'train.npy')
test_set = os.path.join(dataset_dir, 'test.npy')

# this is how we get your final score for the problem.
def evaluation_score(y_pred, y_test):
    y_pred = np.squeeze(y_pred)
    assert y_pred.shape == y_test.shape, "Error: the shape of your prediction doesn't match the shape of ground truth label."

    TP = 0	# truth positive
    FN = 0	# false negetive
    TN = 0	# true negetive
    FP = 0 	# false positive

    for i in range(len(y_pred)):
        pred_label = y_pred[i]
        gt_label = y_test[i]

        if int(pred_label) == -1:
            if pred_label == gt_label:
                TN += 1
            else:
                FN += 1
        else:
            if pred_label == gt_label:
                TP += 1
            else:
                FP += 1

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP) if ((TP + FP) > 0) else 0
    recall = TP / (TP + FN) if ((TP + FN)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if ((precision + recall) > 0) else 0
    final_score = 50 * accuracy + 50 * f1

    return accuracy, precision, recall, f1, final_score

# load dataset
with open(train_set, 'rb') as f:
    X_train = np.load(f)
    y_train = np.load(f)

with open(test_set, 'rb') as f:
    X_test = np.load(f)
    y_test = np.load(f)

clf = BoostingClassifier().fit(X_train, y_train)
print("x train data: ",X_train.shape)
print("y train data:", y_train.shape)
for i in range(X_train.shape[0]):
    print(X_train[i])

# this is how you get your predictions
y_pred = clf.predict(X_test)
acc, precision, recall, f1, final_score = evaluation_score(y_pred, y_test)
print("Accuracy: {}, F-measure: {}, Precision: {}, Recall: {}, Final_Score: {}".format(acc, f1, precision, recall, final_score))

