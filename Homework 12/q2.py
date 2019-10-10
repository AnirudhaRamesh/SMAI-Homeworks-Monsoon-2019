# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

#Part 1
#X in homogeneous form

X_truth = np.array([[1,1,1],[-1,-1,1],[2,2,1],[-2,-2,1],[-1,1,1],[1,-1,1]])
Y_truth = np.array([1,-1,1,-1,1,1])

w_init = np.array([1,0,-1])
idx = np.arange(6)
idx_temp = np.ones(10)
i = 0
#We converged in two iterations on paper
while len(idx_temp)!=0:
    print("w^",i,"=")
    print(w_init)
    error_temp = w_init@X_truth.T
    y_pred = [-1 if x <= 0 else 1 for x in error_temp]
    idx_temp = idx[y_pred!=Y_truth]
    Y_diag = np.diag(Y_truth[idx_temp])
    w_init = w_init + np.sum(Y_diag@X_truth[idx_temp], axis = 0)
    print(idx_temp)
    i+=1
    
    
#Part 2 
print("For part 2")
Y_truth = np.array([-1,-1,1,-1,1,1])

w_init = np.array([1,0,-1])
idx = np.arange(6)
idx_temp = np.ones(10)
i = 0
#We converged in two iterations on paper
while len(idx_temp)!=0 and i < 15:
    print("w^",i,"=")
    print(w_init)
    error_temp = w_init@X_truth.T
    y_pred = [-1 if x <= 0 else 1 for x in error_temp]
    idx_temp = idx[y_pred!=Y_truth]
    Y_diag = np.diag(Y_truth[idx_temp])
    w_init = w_init + np.sum(Y_diag@X_truth[idx_temp], axis = 0)
    print(idx_temp)
    i+=1