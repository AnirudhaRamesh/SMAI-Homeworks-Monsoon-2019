import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def determinant(sig):
    eigen_values, _ = np.linalg.eig(sig)
    res = 0
    for x in eigen_values:
        if x.real > 0:
            res = res + np.log(x.real)
    return res
def log_probability(x, mu, sigma):
    sigma_inv = np.linalg.pinv(sigma)
    first_component = -np.dot(np.dot((x - mu).transpose(), sigma_inv), (x - mu)) / 2
    second_component = -determinant(sigma) / 2
    third_component = -2 * np.log(2 * 22 / 2) / 2
    return (first_component + second_component + third_component)

mu1 = np.array([3,3])
mu2 = np.array([7,7])
def solve(sigma1, sigma2):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(1000):
        sample = np.array([random.random() * 10, random.random() * 10])
        P1 = log_probability(sample, mu1, sigma1)
        P2 = log_probability(sample, mu2, sigma2)
        if P1 > P2:
            x1.append(sample[0])
            y1.append(sample[1])
        else:
            x2.append(sample[0])
            y2.append(sample[1])

    plt.plot(x1, y1, 'o', color = 'r')
    plt.plot(x2, y2, 'o', color = 'g')
    plt.show()
sigma1 = [[3, 0], [0, 3]]
sigma1 = np.array(sigma1)
sigma2 = [[3, 0], [0, 3]]
sigma2 = np.array(sigma2)
solve(sigma1, sigma2)
sigma1 = [[3, 1], [1, 3]]
sigma1 = np.array(sigma1)
sigma2 = [[7, 1], [1, 7]]
sigma2 = np.array(sigma2)
solve(sigma1, sigma2)
