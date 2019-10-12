import numpy as np

data = [[1,1,1], [-1,1,1], [1,-1,1], [-1.3,-1.3,1.3]]
labels = [1, 1, 1, -1]

def margin(weights, datapoints):
    return np.min(np.abs(np.dot(datapoints, weights)) / np.sqrt(weights[0] * weights[0] + weights[1] * weights[1]))


# Perceptron
D = []
for i in range(4):
    D.append(np.array(data[i]))
D = np.array(D)
labels = np.array(labels)
eta = 0.1
w = np.array([1, 1, 0])
c = 0
while True:
    c = c + 1
    flag = 0
    for i in range(4):
        if labels[i] * np.dot(w, D[i]) <= 0:
            flag = 1
            w = w + eta * (labels[i] - np.sign(np.dot(w, D[i]))) * D[i]
    if flag == 0:
        break
print("Weights for Perceptron Algorithm = ", w)
print("Margin for Perceptron = ", str(margin(w, D)))
pred = []
for i in range(4):
    if np.dot(w, D[i]) < 0:
        pred.append(-1)
    else:
        pred.append(1)
print("Predictions for Perceptron Algorithm = ", pred)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def predict(X, w, gamma):
    if sigmoid(gamma * np.dot(X, w)) <= 0.5:
        return -1
    else:
        return 1


def logistic_regression(train_data, train_pred, gamma):
    w = np.ones(train_data.shape[1])
    prev_w = np.zeros(train_data.shape[1])
    cnt = 1
    while np.linalg.norm(prev_w - w) > 0.001:
        # Do Gradient Descent
        deltaJ = np.zeros(train_data.shape[1])
        for i in range(train_data.shape[1]):
            res = 0
            for k in range(train_data.shape[0]):
                den = 1 + np.exp(-train_pred[k] * gamma * np.dot(train_data[k], w))
                num = -(np.exp(-gamma * train_pred[k] * np.dot(train_data[k],w))) * (train_pred[k] * gamma * train_data[k][i])
                res = res + (num / den)
            deltaJ[i] = res
        eta = 0.1
        prev_w = w
        w = w - eta * deltaJ
        cnt = cnt + 1
    print("Iterations to converge for LR with gamma value %s = " %(str(gamma)), cnt)
    return w

gammas = [0.1, 0.2, 0.5, 0.8, 1, 2, 5, 10, 20, 50, 100]
for g in gammas:
    wt = logistic_regression(D, labels, g)
    print("Weights for LR with gamma value %s = " %(str(g)), wt)
    print("Margin for LR with gamma value %s = " % (str(g)), margin(wt, D))
    pred = []
    for j in range(4):
        pred.append(predict(D[j], wt, g))
    print("Predictions for LR with gamma value %s = " % (str(g)), pred)