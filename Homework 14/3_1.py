import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X_train,X_test,Y_train,Y_test = train_test_split(X, Y,test_size=1/7)

    X_2_train = np.vstack((X_train[Y_train=='1'][:500],X_train[Y_train=='2'][:500]))
    X_2_test = np.vstack((X_test[Y_test=='1'],X_test[Y_test=='2']))
    Y_2_train = np.hstack((Y_train[Y_train=='1'][:500],Y_train[Y_train=='2'][:500]))
    Y_2_test = np.hstack((Y_test[Y_test=='1'],Y_test[Y_test=='2']))

    clf = SVC(C=1,max_iter = 10000, kernel='linear')
    clf.fit(X_2_train,Y_2_train)

    H = clf.predict(X_2_test)
    Accuracy = np.sum(H == Y_2_test) / Y_2_test.size
    print("Accuracy = ", Accuracy * 100, '%')

    pca = PCA(n_components=2)
    pca.fit(X_2_test)
    X_2_test_projection = pca.transform(X_2_test)

    W = clf.coef_[0]
    B = clf.intercept_
    U = pca.components_
    W_project = U.dot(W)
    x = np.linspace(-100,400,500)
    y = -(W_project[0]*x + B)/W_project[1]

    plt.scatter(X_2_test_projection[Y_2_test=='1'][:,0],X_2_test_projection[Y_2_test=='1'][:,1])
    plt.scatter(X_2_test_projection[Y_2_test=='2'][:,0],X_2_test_projection[Y_2_test=='2'][:,1])
    plt.plot(x,y,'black',linewidth=5)
    plt.show()

    V = clf.support_vectors_
    V_project = V.dot(U.T)
    # plt.scatter(X_2_test_projection[Y_2_test=='1'][:,0],X_2_test_projection[Y_2_test=='1'][:,1])
    # plt.scatter(X_2_test_projection[Y_2_test=='2'][:,0],X_2_test_projection[Y_2_test=='2'][:,1])
    plt.scatter(V_project[:,0],V_project[:,1])
    plt.plot(x,y,'black',linewidth=5)
    plt.show()
