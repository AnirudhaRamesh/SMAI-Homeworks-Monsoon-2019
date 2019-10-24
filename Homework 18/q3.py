import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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

if __name__ == '__main__':

    X_train, Y_train = read_data("sample_train.csv")
    X_test, Y_test = read_data("sample_test.csv")
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)

    X_2_train = np.vstack((X_train[Y_train==1],X_train[Y_train==2]))
    X_2_test = np.vstack((X_test[Y_test==1],X_test[Y_test==2]))
    Y_2_train = np.hstack((Y_train[Y_train==1],Y_train[Y_train==2]))
    Y_2_test = np.hstack((Y_test[Y_test==1],Y_test[Y_test==2]))

    # print(X_2_train.shape, Y_2_train.shape)
    # print(X_2_test.shape, Y_2_test.shape)

    scaler = StandardScaler()
    scaler.fit(X_2_train)
    X_2_train = scaler.transform(X_2_train)
    X_2_test = scaler.transform(X_2_test)

    # Sklearn neural_network
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000,1000), random_state=42,activation='relu')
    clf.fit(X_2_train, Y_2_train)
    H = clf.predict(X_2_test)

    Accuracy = np.sum(H == Y_2_test) / Y_2_test.size
    print("Accuracy = ", Accuracy * 100, '%')
