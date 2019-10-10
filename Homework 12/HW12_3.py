# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:54:45 2019

@author: aryan
"""
#using boston dataset adjusted for inflation
#using only 4 relevant data points

import numpy as np
import matplotlib.pyplot as plt

#Function to Calculate del J:
def grad_w(X,Y,w):
    return X.T@(X@w-Y)


#Function to Calculate Hessian:
def Hessian(X):
    H_mat = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            H_mat[i,j] = np.matmul(X[:,i], X[:,j])
    return H_mat
            
            
def error(X,Y,w):
    return np.linalg.norm(X@w-Y)

            
my_data = np.genfromtxt('housing.csv', delimiter=',')
XOG = np.ones((my_data.shape[0]-1,4))
XOG[:,0:3] = my_data[1:,0:3]

X = np.ones((my_data.shape[0]-1,4))
X[:,0:3] = my_data[1:,0:3]
X - np.mean(X, axis=0)

Y_original = np.ones((489,1))
Y_original = my_data[1:,3]
Y = np.ones((489,1))
Y = (my_data[1:,3] - np.mean(Y_original))/(np.max(Y_original) - np.min(Y_original))
Y.shape = [489,1]
w = np.zeros((4,1))



#Normal Gradient Descent with 0.5 learning rate
#1000 iterations


    

x_arr = []
for i in range(10000):
    w = w - 0.000005*grad_w(X,Y,w)
    err = error(X,Y,w)
    if err < 2.15:
        break
    x_arr.append(error(X,Y,w))
    
plt.plot(x_arr)
plt.show()





#gradient Descent with method 2:

   

x_arr = []
err = 10000000
w = np.zeros((4,1))

for i in range(10000):
    J = grad_w(X,Y,w)
    J.shape = [4,1]
    H = Hessian(X)
    n = np.linalg.norm(J)*np.linalg.norm(J)/(J.T@H@J)
    w = w - n*grad_w(X,Y,w)
    err = error(X,Y,w)
    x_arr.append(error(X,Y,w))
    if err < 2.15:
        break
    
    
plt.plot(x_arr)
plt.show()


#method 3
w = np.zeros((4,1))
x_arr = []
err = 10000000
w = np.zeros((4,1))

for i in range(10000):
    J = grad_w(X,Y,w)
    J.shape = [4,1]
    H = Hessian(X)
    n = np.linalg.norm(J)*np.linalg.norm(J)/(J.T@H@J)
    w = w - np.linalg.inv(H)@J
    err = error(X,Y,w)
    x_arr.append(error(X,Y,w))
    if err < 2.15:
        break
    
    
plt.plot(x_arr)
plt.show()