# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import numpy as np
from sklearn.datasets import load_boston
import math
import random
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)

    for j,tau in enumerate(taus):
        M = np.exp(-1 * l2(x_test, x_train) / (2*(tau**2)))
        # M = -1 * l2(x_test, x_train) / (2*(tau**2))
        predictions =  np.array([LRLS(x_test[i,:],x_train,y_train, tau, M[i,:]) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
        print(j+1, "/", len(taus))

    return losses

#to implement
def LRLS(test_datum,x_train,y_train, tau,M,lam=1e-5):
    '''
    Input: test_data is a 1xd test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           M is a 1*N_train matrix
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    den = np.sum(M)
    # den = logsumexp(M)
    A = np.zeros((x_train.shape[0], x_train.shape[0]))
    for i in range(x_train.shape[0]):
        A[i][i] = M[i] / den
        # A[i][i] = np.exp(M[i] - den)
    # left,right means the terms in the corresponding sides in analytical solution
    left = np.dot(np.dot(x_train.transpose(), A),  x_train) + lam * np.eye(x_train.shape[1])
    right = np.dot(np.dot(x_train.transpose(), A), y_train)
    w_opt = np.linalg.solve(left, right)
    y_pred = np.dot(test_datum, w_opt)
    return y_pred



def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''

    subset_size = len(x)/k
    random_indices = np.random.permutation(range(len(x)))# Generate shuffled indices into dataset
    losses = np.zeros((k,len(taus)))

    for i in range(k):
        print('The program is under', i+1, 'round. It runs', k, 'rounds in total.')
        test_indices = random_indices[i*subset_size: (i+1)*subset_size]
        train_indices = [k for k in random_indices if k not in test_indices]# Remove test set from the whole set to form training set
        losses[i,:] = run_on_fold(x[test_indices,:], y[test_indices], \
            x[train_indices,:], y[train_indices], taus)
    losses = np.mean(losses,axis=0)
    return losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)

    plt.scatter(taus,losses)
    plt.xlabel(r'$\tau$')
    plt.ylabel('losses')
    print("min loss = {}".format(losses.min()))
    plt.show()
