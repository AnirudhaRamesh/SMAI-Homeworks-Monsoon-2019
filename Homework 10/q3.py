import numpy as np
import matplotlib.pyplot as plt


""" Load data from a single batch """

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

d = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')
data = d[b'data']
labels = d[b'labels']
print("Shape of the data: " + str(data.shape))
print("Length of labels list = " + str(len(labels)))

####### Restructure Data ########
D = []
for i in range(10):
    D.append([])

for i in range(len(labels)):
    D[labels[i]].append((data[i, :1024] * 0.299 + data[i, 1024:2048] * 0.587 + data[i, 2048:3072] * 0.114) / 255)
for i in range(10):
    D[i] = np.array(D[i])
    D[i] = D[i].astype(np.float64)
D = np.array(D)
# Code for Part 1
means = []
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ys = []
for i in range(10):
    means.append(np.mean(D[i], axis = 0))
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(D[i].T))
    np.flip(eigen_values.argsort())
    E = np.dot(D[i], eigen_vectors[:, 0:20])
    reconstruct = np.dot(E, eigen_vectors[:, 0:20].T).real
    diff = (reconstruct - D[i]) ** 2
    tmp = np.sqrt(np.sum(diff, axis = 1))
    error = np.sum(tmp) / D[i].shape[0]
    ys.append(error)
plt.plot(x, ys, 'r')
plt.show()
# Code for Part 2
print("\n\n\nMATRIX OF PAIRWISE MEAN DISTANCES\n\n")
for i in range(10):
    t = []
    for j in range(10):
        print('{:19}'.format(np.linalg.norm(means[i] - means[j])), end = " ")
    print()

# Code for Part 3
mat = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        Di = D[i] - means[i]
        eigen_valuesi, eigen_vectorsi = np.linalg.eig(np.cov(Di.T))
        np.flip(eigen_valuesi.argsort())
        E = np.dot(Di, eigen_vectorsi[:, 0:20])
        Dj = D[j] - means[j]
        eigen_valuesj, eigen_vectorsj = np.linalg.eig(np.cov(Dj.T))
        reconstruct = np.dot(E, eigen_vectorsj[:, 0:20].T) + means[i]
        reconstruct = reconstruct.real
        diff = (reconstruct - D[i])
        diff = diff ** 2
        tmp = np.sqrt(np.sum(diff, axis = 1))
        error = np.sum(tmp) / reconstruct.shape[0]
        mat[i][j] = error

print("\n\n\n\n\n")
for i in range(10):
    idx1 = -1
    idx2 = -1
    idx3 = -1
    for _ in range(3):
        sm = 100000000000000000000000000000
        res = -1
        for j in range(10):
            if j == idx1 or j == idx2 or j == idx3 or j == i:
                continue
            ttt = (mat[i][j] + mat[j][i]) * 0.5
            if ttt < sm:
                sm = ttt
                res = j
        if _ == 0:
            idx1 = res
        elif _ == 1:
            idx2 = res
        elif _ == 2:
            idx3 = res
    print("Class" + str(i) + ": " + str(idx1) + " " + str(idx2) + " " + str(idx3))
