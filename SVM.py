# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 08:51:40 2015

Implementation of a SVM

@author: ciromontanez
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt.solvers import qp

# read Xa and ya to train
preXa = np.fromfile("p1_a_X.dat", dtype = float, sep='\t')
Xa = preXa.reshape(2000,2)

ya = np.fromfile("p1_a_y.dat", dtype = float, sep='\t')

# read Xb and yb to train
preXb = np.fromfile("p1_b_X.dat", dtype = float, sep='\t')
Xb = preXb.reshape(2000,2)

yb = np.fromfile("p1_b_y.dat", dtype = float, sep='\t')

def SVM_train(X,y):
    # get X dimensions
    n, d = np.shape(X)
    # define matrices to solve quadratic program
    # as especified in cvxopt.solvers.qp
    preP = np.identity(d+1)
    preP[d,d]=0
    preq = np.zeros((d+1,1))
    preG = np.append(X, np.ones((n,1)), axis = 1)
    preG = -1 * preG * y[:,np.newaxis]
    preh = -1 * np.ones((n,1))
    # convert all to matrix in cvxopt
    P = matrix(preP)
    q = matrix(preq)
    G = matrix(preG)
    h = matrix(preh)
    # optimize using qp from 
    theta = qp(P, q, G, h)['x']  
    return theta
    
theta = SVM_train(Xb,yb)
    
# plot boundary
def plot_boundary(theta, X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0][y==1], X[:,1][y==1], c='b')
    ax.scatter(X[:,0][y==-1], X[:,1][y==-1], c='g')
    xline = np.arange(20,180)
    yline = -(xline*theta[0]+theta[2])/theta[1]
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
    
marginAndBound((theta[0], theta[1]), Xa, ya)

# calculate geometric margin alternative
def geometricMargin(theta, X):
    normTheta = np.linalg.norm(theta)
    geomargin = min([abs(np.dot(theta,x)) for x in X])/normTheta
    return geomargin 
    
geometricMargin((theta[0], theta[1]), Xa)

