from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

""" LOAD DATA """
def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")

print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

""" Train 10C2 Binary Classifiers """
classifiers = []
for i in range(10):
    for j in range(10):
        X = []
        Y = []
        if i == j:
            continue
        for k in range((train_data.shape[0])):
            if int(train_labels[k]) == i:
                X.append(train_data[k])
                Y.append(1)
            if int(train_labels[k]) == j:
                X.append(train_data[k])
                Y.append(-1)
        clf = LogisticRegression(solver = 'liblinear', max_iter = 100000).fit(X, Y)
        classifiers.append(clf)

""" Computing Testing accuracy """
acc = 0
predictions = []
for k in range(len(test_data)):
    c = 0
    freq = np.zeros(10)
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            pred = classifiers[c].predict(test_data[k].reshape(1, -1))
            if pred[0] == 1:
                freq[i] += 1
            elif pred[0] == -1:
                freq[j] += 1
            else:
                assert(False)
            c = c + 1
    max_f = 0
    it = -1
    for i in range(10):
        if freq[i] > max_f:
            max_f = freq[i]
            it = i
    predictions.append(it)
    if it == test_labels[k]:
        acc = acc + 1
print("Accuracy = ", str(acc / len(test_data) * 100), "%")

""" Print Confusion Matrix """
from sklearn import metrics
cm = metrics.confusion_matrix(test_labels, predictions)
print("\n\nConfusion Matrix:")
print(cm)