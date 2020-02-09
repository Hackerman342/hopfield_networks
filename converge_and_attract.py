# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: kwc57
"""

# Import package dependencies
import math
import numpy as np 
import matplotlib.pyplot as plt

# Initialize memory patterns
x1 = np.array([-1., -1., 1., -1., 1., -1., -1., 1.]).reshape(1,8)
x2 = np.array([-1., -1., -1., -1., -1., 1., -1., -1.]).reshape(1,8)
x3 = np.array([-1., 1., 1., -1., -1., 1., -1., 1.]).reshape(1,8)

# Calculate Weight Matrix (Scaling optional)
W = (1./8.)*(np.dot(x1.T,x1) + np.dot(x2.T,x2) + np.dot(x3.T,x3))

# Display weight matrix as greyscale image
plt.imshow(W,  cmap='gray')
plt.show()

#pict = np.loadtxt('pict.dat')
#print(pict.shape)

x1_recall = np.sign(np.dot(W,x1.T).T)
x2_recall = np.sign(np.dot(W,x2.T).T)
x3_recall = np.sign(np.dot(W,x3.T).T)

# Plot recalls over inputs to ensure proper weight matrix configuration
#plt.title("Confirming proper x1 recall")
#plt.plot(x1[0], label='Input')
#plt.plot(x1_recall[0], label='Recall')
#plt.legend()
#plt.show()
#
#plt.title("Confirming proper x2 recall")
#plt.plot(x2[0], label='Input')
#plt.plot(x2_recall[0], label='Recall')
#plt.legend()
#plt.show()
#
#plt.title("Confirming proper x3 recall")
#plt.plot(x3[0], label='Input')
#plt.plot(x3_recall[0], label='Recall')
#plt.legend()
#plt.show()


# Initialize distorted memory patterns
x1d = np.array([1., -1., 1., -1., 1., -1., -1., 1.]).reshape(1,8)
x2d = np.array([1., 1., -1., -1., -1., 1., -1., -1.]).reshape(1,8)
x3d = np.array([1., 1., 1., -1., 1., 1., -1., 1.]).reshape(1,8)

for i in range(3):
    x1d = np.sign(np.dot(W,x1d.T).T)
    x2d = np.sign(np.dot(W,x2d.T).T)
    x3d = np.sign(np.dot(W,x3d.T).T)
    
# Plot recalls over inputs to ensure proper weight matrix configuration
plt.title("Checking x1d convergence")
plt.plot(x1[0], label='Attractor Input')
plt.plot(x1d[0], label='Distorted Recall')
plt.legend()
plt.show()

plt.title("Checking x2d convergence")
plt.plot(x2[0], label='Attractor Input')
plt.plot(x2d[0], label='Distorted Recall')
plt.legend()
plt.show()

plt.title("Checking x3d convergence")
plt.plot(x3[0], label='Attractor Input')
plt.plot(x3d[0], label='Distorted Recall')
plt.legend()
plt.show()
