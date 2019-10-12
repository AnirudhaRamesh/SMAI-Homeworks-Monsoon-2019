import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression_loss(W,X,Y):
    return np.sum(np.square(sigmoid(X.dot(W)) - Y)) / Y.size

def linear_regression_loss(W,X,Y):
    return np.sum(np.square(X.dot(W) - Y)) / Y.size

if __name__ == '__main__':
    N = 1000
    points_a = np.random.normal(1,1,(N,1))
    points_b = np.random.normal(-1,1,(N,1))

    X = np.vstack((points_a,points_b))
    X = np.c_[X,np.ones(X.shape[0])]
    Y = np.hstack((np.ones(N),-np.ones(N)))

    # print(X.shape)
    # print(Y.shape)

    M = 100
    x = np.linspace(-20,20,M)
    xx,yy = np.meshgrid(x,x)

    Z1 = np.zeros((M,M))
    Z2 = np.zeros((M,M))

    for i in range(M):
        for j in range(M):
            W = np.array([i,j])
            Z1[i][j] = linear_regression_loss(W,X,Y)
            Z2[i][j] = logistic_regression_loss(W,X,Y)

    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(xx,yy,Z1)
    ax1.set_xlabel('W1')
    ax1.set_ylabel('W0')
    ax1.set_zlabel('Loss')
    ax1.set_title("Linear Regression Loss Function")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(xx,yy,Z2)
    ax2.set_xlabel('W1')
    ax2.set_ylabel('W0')
    ax2.set_zlabel('Loss')
    ax2.set_title("Logistic Regression Loss Function")

    plt.show()
