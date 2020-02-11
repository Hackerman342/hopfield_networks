#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020
@author: kwc57
"""

# Import package dependencies
import math
import numpy as np 
import matplotlib.pyplot as plt
from itertools import permutations


# Initialize memory patterns
x1 = np.array([-1., -1., 1., -1., 1., -1., -1., 1.]).reshape(1,8)
x2 = np.array([-1., -1., -1., -1., -1., 1., -1., -1.]).reshape(1,8)
x3 = np.array([-1., 1., 1., -1., -1., 1., -1., 1.]).reshape(1,8)

# Calculate Weight Matrix (Scaling optional)
W = (1./8.)*(np.dot(x1.T,x1) + np.dot(x2.T,x2) + np.dot(x3.T,x3))
np.fill_diagonal(W,0)
# Display weight matrix as greyscale image
plt.imshow(W,  cmap='gray')
plt.colorbar()
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

<<<<<<< HEAD:converge_and_attract_kyle
for i in range(5):
=======
for i in range(3):
>>>>>>> c7f47a846eca2c1cce9bced56951b10b5866cc18:converge_and_attract_kyle.py
    x1d = np.sign(np.dot(W,x1d.T).T)
    x2d = np.sign(np.dot(W,x2d.T).T)
    x3d = np.sign(np.dot(W,x3d.T).T)
    
## Plot recalls over inputs to ensure proper weight matrix configuration
#plt.title("Checking x1d convergence")
#plt.plot(x1[0], "c", label='Attractor Input')
#plt.plot(x1d[0], "--k", label='Distorted Recall')
#plt.legend()
#plt.show()
#
#plt.title("Checking x2d convergence")
#plt.plot(x2[0], "c", label='Attractor Input')
#plt.plot(x2d[0], "--k", label='Distorted Recall')
#plt.legend()
#plt.show()
#
#plt.title("Checking x3d convergence")
#plt.plot(x3[0], "c", label='Attractor Input')
#plt.plot(x3d[0], "--k", label='Distorted Recall')
#plt.legend()
#plt.show()

plt.title("Displaying x1")
plt.imshow(x1,  cmap='gray')
plt.colorbar()
plt.show()

plt.title("Checking x1d convergence")
plt.imshow(x1d,  cmap='gray')
plt.colorbar()
plt.show()

plt.title("Displaying x2")
plt.imshow(x2,  cmap='gray')
plt.colorbar()
plt.show()

plt.title("Checking x2d convergence")
plt.imshow(x2d,  cmap='gray')
plt.colorbar()
plt.show()

plt.title("Displaying x3")
plt.imshow(x3,  cmap='gray')
plt.colorbar()
plt.show()

plt.title("Checking x3d convergence")
plt.imshow(x3d,  cmap='gray')
plt.colorbar()
plt.show()

##### Automate the search for attractors ######
a0 = [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,   1.]
a1 = [-1.,  1.,  1.,  1.,  1.,  1.,  1.,   1.]
a2 = [-1., -1.,  1.,  1.,  1.,  1.,  1.,   1.]
a3 = [-1., -1., -1.,  1.,  1.,  1.,  1.,   1.]
a4 = [-1., -1., -1., -1.,  1.,  1.,  1.,   1.]
a5 = [-1., -1., -1., -1., -1.,  1.,  1.,   1.]
a6 = [-1., -1., -1., -1., -1., -1.,  1.,   1.]
a7 = [-1., -1., -1., -1., -1., -1., -1.,   1.]
a8 = [-1., -1., -1., -1., -1., -1., -1.,   1.]
a9 = [-1., -1., -1., -1., -1., -1., -1.,  -1.] 

perm0 = set(permutations(a0))
perm1 = set(permutations(a1))
perm2 = set(permutations(a2))
perm3 = set(permutations(a3))
perm4 = set(permutations(a4))
perm5 = set(permutations(a5))
perm6 = set(permutations(a6))
perm7 = set(permutations(a7))
perm8 = set(permutations(a8))
perm9 = set(permutations(a9))

ar0 = np.array(list(perm0))
ar1 = np.array(list(perm1))
ar2 = np.array(list(perm2))
ar3 = np.array(list(perm3))
ar4 = np.array(list(perm4))
ar5 = np.array(list(perm5))
ar6 = np.array(list(perm6))
ar7 = np.array(list(perm7))
ar8 = np.array(list(perm8))
ar9 = np.array(list(perm9))

all_pos = np.concatenate((ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9), axis=0)

## Display all possible inputs as greyscale image
#plt.imshow(all_pos,  cmap='gray')
#plt.show()


# Takes 3 iterations for all inputs to reach an attractor (2 ~= ln(8))
for i in range(3):
    all_pos = np.sign(np.dot(W,all_pos.T).T)
    
attractors = np.unique(all_pos, axis=0)
print("\n\n Number of attractors (assuming convergence): ", attractors.shape[0])

# Display all attractors as greyscale image
plt.title("All attractors")
plt.imshow(attractors,  cmap='gray')
plt.colorbar()
plt.show()

