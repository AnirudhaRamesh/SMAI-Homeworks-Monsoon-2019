import numpy as np
import matplotlib.pyplot as plt
import csv

f = open('wine.data', 'r')
csvreader = csv.reader(f)
data = []
labels = []
for rows in csvreader:
    for i in range(len(rows)):
        rows[i] = float(rows[i])
    labels.append(int(rows[0]))
    data.append(np.array(rows[1:]).astype(np.float64))
data = np.array(data)
labels = np.array(labels)
data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)

# Plot for eigenvalues

eigen_values, eigen_vectors = np.linalg.eig(np.cov(data.T))
idx = np.flip(eigen_values.argsort())
plt.plot(eigen_values[idx], 'ro')
plt.show()

proj_data = np.dot(data, eigen_vectors[:, 0:2])
class1 = []
class2 = []
class3 = []
for i in range(len(labels)):
    if labels[i] == 1:
        class1.append(proj_data[i])
    elif labels[i] == 2:
        class2.append(proj_data[i])
    elif labels[i] == 3:
        class3.append(proj_data[i])
    else:
        assert(False)
class1 = np.array(class1)
class2 = np.array(class2)
class3 = np.array(class3)
plt.plot(class1[:, 0], class1[:, 1], 'ro', label = "Class 1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label = "Class 2")
plt.plot(class3[:, 0], class3[:, 1], 'go', label = "Class 3")
plt.gca().legend()
plt.show()