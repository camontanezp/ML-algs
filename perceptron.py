# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:46:53 2015

Implementation of the perceptron algorithm

@author: ciromontanez
"""
import numpy as np
import matplotlib.pyplot as plt

# read Xa and ya to train
preXa = np.fromfile("p1_a_X.dat", dtype = float, sep='\t')
Xa = preXa.reshape(2000,2)

ya = np.fromfile("p1_a_y.dat", dtype = float, sep='\t')

# read Xb and yb to train
preXb = np.fromfile("p1_b_X.dat", dtype = float, sep='\t')
Xb = preXb.reshape(2000,2)

yb = np.fromfile("p1_b_y.dat", dtype = float, sep='\t')


# plot of Xa and Xb
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.scatter(Xa[:,0][ya==1], Xa[:,1][ya==1], c='b')
ax1.scatter(Xa[:,0][ya==-1], Xa[:,1][ya==-1], c='g')
ax1.set_title('Xa')
ax2.scatter(Xb[:,0][yb==1], Xb[:,1][yb==1], c='b')
ax2.scatter(Xb[:,0][yb==-1], Xb[:,1][yb==-1], c='g')
ax2.set_title('Xb')
plt.show()  

def perceptron_train(X, y):
    """
    This function trains a Perceptron classifier
    on a training set of n examples of dimension d.
    The labels of the exmaples ar in y and are 1 and -1
    
    @param: X of dim nxd
    @param: y of dim nx1
    
    @return: (theta, k) the final classification vector 
    and the number of updates performed.
    """

    # dimension of each example
    d = np.shape(X)[1]
    # number of training examples
    n = np.shape(X)[0]
    # define and initialize theta
    theta = [0]*d
    # define and initialize k to 0
    k = 0
    
    # for each training example x_t modify theta
    # so that y_t*theta*x_t becomes positive
    for i in range(n):
        
        # do until y_t*theta*x_t becomes positive 
        while y[i] != f(X[i], theta):
            theta = theta + y[i]*X[i]
            k += 1
    
    return (theta, k)
        
    
# definition of the linear classifier f
def f(x, theta):
    return np.sign(np.dot(x, theta))

# definition of the perceptron test
def perceptron_test(theta, X, y):
    """
    Calculates the error of a classifier on a 
    given test data set
    
    @param: theta is the classification vector
    @param: X is matrix containing the 
    vectros that will be classified. Size mxd
    @param: y is the vector of true labels
    for X. Size dx1
    
    @return: errors/m the proportion of errors
    """
    # number of test vectors
    m = np.shape(X)[0]
    # errors to store the number of wrong classifications
    errors = 0
    
    # count errors
    for i in range(m):
        if f(X[i], theta) != y[i]:
            errors += 1
    
    # return fraction of errors
    return errors/m
    
# train
(theta, k) = perceptron_train(Xa, ya)

# test
test_err = perceptron_test(theta, Xb, yb)

# plot boundary
def plot_boundary(theta, X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0][y==1], X[:,1][y==1], c='b')
    ax.scatter(X[:,0][y==-1], X[:,1][y==-1], c='g')
    xline = np.arange(20,180)
    yline = -xline*theta[0]/theta[1]
    ax.plot(xline, yline, 'r')
    plt.show()
        
plot_boundary(theta, Xa, ya)

# calculate the geometric margin and bound on steps
def marginAndBound(theta, X, y):
    gamma = min([np.dot(theta,X[i])*y[i] for i in range(len(y))])
    gammaGeom = gamma/np.linalg.norm(theta)
    r = max([np.linalg.norm(x) for x in X])
    bound = (r/gammaGeom)**2
    return (gamma, gammaGeom, bound)

# calculate geometric margin alternative
def geometricMargin(theta, X):
    normTheta = np.linalg.norm(theta)
    geomargin = min([abs(np.dot(theta,x)) for x in X])/normTheta
    return geomargin 
    

    
    
    
    
    