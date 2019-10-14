# cook your dish here
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import math

def solve(k_folds, data, noise, title):
    ks = []
    means = []
    Vars = []
    for k in k_folds:
	    kfold = KFold(k, True, 1)
	    fold_means = []
	    for train, test in kfold.split(data):
		    sample_errs = [noise[i] for i in data[train]]
		    fold_means.append(np.mean(np.array(sample_errs)))
	    mean = np.mean(fold_means)
	    var = np.var(fold_means)
	    Vars.append(var)
	    ks.append(k)
	    means.append(mean)
    p1 = ks
    p2 = Vars
    p3 = means
    fig = plt.figure()
    plt.title(title)
    ax1 = fig.add_subplot(121)
    ax1.plot(p1, p2, '*', linestyle = '--', color = 'b')
    ax1.set_ylabel('Error Variance')
    ax1.set_xlabel('Number of folds(K)')
    ax2 = fig.add_subplot(122)
    ax2.plot(p1, p3,'*', linestyle = '--', color = 'b')
    ax2.set_ylabel('Error Mean')
    ax2.set_xlabel('Number of folds(K)')
    plt.show()

x = np.array(list(range(100)))
# Gaussian Noise
noise = np.random.normal(0, 1, 100)
folds = [2, 4, 5, 10, 20, 25, 50]
solve(folds, x, noise, '100 samples')

x = np.array(list(range(10000)))
# Gaussian Noise
noise = np.random.normal(0, 1, 10000)
folds = [2, 4, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
solve(folds, x, noise, '10000 samples')