from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

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

train_data = (train_data - np.mean(train_data, axis = 0))
# NORMAL PCA
eigen_values, eigen_vectors = np.linalg.eig(np.cov(train_data.T))
np.flip(eigen_values.argsort())
plt.scatter(eigen_vectors[:, 0].real.dot(train_data.T), eigen_vectors[:, 1].real.dot(train_data.T))
plt.title("NORMAL PCA")
plt.show()
projected = np.zeros((train_data.shape[0], 2))
projected[:, 0] = np.dot(train_data, eigen_vectors[:, 0].real) 
projected[:, 1] = np.dot(train_data, eigen_vectors[:, 1].real)

eig_inv = []
eig_inv.append(eigen_vectors[:, 0].real)
eig_inv.append(eigen_vectors[:, 1].real)
eig_inv = np.array(eig_inv)

reconstruct = np.dot(projected, eig_inv)
print(np.linalg.norm(reconstruct - train_data))

# GRADIENT DESCENT PCA without regularization

""" Compute w """
w1 = np.random.rand(784)
learning_rate = 0.1
prev_w = np.zeros(784)
for i in range(1000):
    prev_w = w1
    w1 = w1 + learning_rate * np.dot(np.dot(train_data.T, train_data) , w1)
    w1 = w1 / np.linalg.norm(w1)
    if np.linalg.norm(prev_w - w1) < 0.00001:
        break

""" Compute w dash """
w2 = np.random.rand(784)
prev_w = np.zeros(784)
for i in range(1000):
    prev_w = w2
    tmp = train_data - np.dot(np.matmul(train_data, w1).reshape(-1, 1), w1.reshape(1, 784))
    w2 = w2 + learning_rate * np.dot(np.dot(tmp.T, tmp) , w2)
    w2 = w2 / np.linalg.norm(w2)
    if np.linalg.norm(prev_w - w2) < 0.00001:
        break
plt.scatter(np.dot(train_data, w1), np.dot(train_data, w2))
plt.title("GRADIENT DESCENT PCA NO REGULARIZATION")
plt.show()
projected = np.zeros((train_data.shape[0], 2))
projected[:, 0] = np.dot(train_data, w1) 
projected[:, 1] = np.dot(train_data, w2)

eig_inv = []
eig_inv.append(w1)
eig_inv.append(w2)
eig_inv = np.array(eig_inv)

reconstruct = np.dot(projected, eig_inv)
print(np.linalg.norm(reconstruct - train_data))

# GRADIENT DESCENT PCA with L1 regularization
reg = 5000
""" Compute w """
w1 = np.random.rand(784)
learning_rate = 0.1
prev_w = np.zeros(784)
for i in range(1000):
    prev_w = w1
    grad = np.zeros(784)
    for i in range(784):
        if w1[i] > 0:
            grad[i] = 1
        elif w1[i] < 0:
            grad[i] = -1
        else:
            grad[i] = 0
    w1 = w1 + learning_rate * (np.dot(np.dot(train_data.T, train_data) , w1) + reg * grad)
    w1 = w1 / np.linalg.norm(w1)
    if np.linalg.norm(prev_w - w1) < 0.00001:
        break

""" Compute w dash """
w2 = np.random.rand(784)
prev_w = np.zeros(784)
for i in range(10000):
    prev_w = w2
    tmp = train_data - np.dot(np.matmul(train_data, w1).reshape(-1, 1), w1.reshape(1, 784))
    grad = np.zeros(784)
    for i in range(784):
        if w2[i] > 0:
            grad[i] = 1
        elif w2[i] < 0:
            grad[i] = -1
        else:
            grad[i] = 0
    w2 = w2 + learning_rate * (np.dot(np.dot(tmp.T, tmp) , w2) + reg * grad)
    w2 = w2 / np.linalg.norm(w2)
    if np.linalg.norm(prev_w - w2) < 0.00001:
        break
plt.scatter(np.dot(train_data, w1), np.dot(train_data, w2))
plt.title("GRADIENT DESCENT PCA L1 REGULARIZATION")
plt.show()
projected = np.zeros((train_data.shape[0], 2))
projected[:, 0] = np.dot(train_data, w1) 
projected[:, 1] = np.dot(train_data, w2)

eig_inv = []
eig_inv.append(w1)
eig_inv.append(w2)
eig_inv = np.array(eig_inv)

reconstruct = np.dot(projected, eig_inv)
print(np.linalg.norm(reconstruct - train_data))

# GRADIENT DESCENT PCA with L2 regularization

""" Compute w """
reg = 2 * (10 ** 7)
w1 = np.random.rand(784)
learning_rate = 0.1
prev_w = np.zeros(784)
for i in range(10000):
    prev_w = w1
    w1 = w1 + learning_rate * (np.dot(np.dot(train_data.T, train_data) , w1) + reg * w1)
    w1 = w1 / np.linalg.norm(w1)
    if np.linalg.norm(prev_w - w1) < 0.00001:
        break

""" Compute w dash """
w2 = np.random.rand(784)
prev_w = np.zeros(784)
for i in range(1000):
    prev_w = w2
    tmp = train_data - np.dot(np.matmul(train_data, w1).reshape(-1, 1), w1.reshape(1, 784))
    w2 = w2 + learning_rate * (np.dot(np.dot(tmp.T, tmp) , w2) + reg * w2)
    w2 = w2 / np.linalg.norm(w2)
    if np.linalg.norm(prev_w - w2) < 0.00001:
        break
plt.scatter(np.dot(train_data, w1), np.dot(train_data, w2))
plt.title("GRADIENT DESCENT PCA L2 REGULARIZATION")
plt.show()
projected = np.zeros((train_data.shape[0], 2))
projected[:, 0] = np.dot(train_data, w1) 
projected[:, 1] = np.dot(train_data, w2)

eig_inv = []
eig_inv.append(w1)
eig_inv.append(w2)
eig_inv = np.array(eig_inv)

reconstruct = np.dot(projected, eig_inv)
print(np.linalg.norm(reconstruct - train_data))