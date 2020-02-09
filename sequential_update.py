# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: kwc57
"""

# Import package dependencies
import math
import numpy as np 
import matplotlib.pyplot as plt

# Import necessary classes from script with functions
import hopfield_functions as hf

# Load data from pict
pict = np.genfromtxt('pict.dat', delimiter=',').reshape(-1,1024)

# Show images being used for training
show_images = False
if show_images:
    for i in range(pict.shape[0]):
        img = pict[i].reshape(int(math.sqrt(pict.shape[1])),-1)
        plt.imshow(img,  cmap='gray')
        plt.show()

# Calculate Weight Matrix (Scaling optional)
W = hf.weight_calc(pict,3)

# Recall training images
pic1_recall = np.sign(np.dot(W,pict[0].T).T)
pic2_recall = np.sign(np.dot(W,pict[1].T).T)
pic3_recall = np.sign(np.dot(W,pict[2].T).T)

# Plot input images and recalls
plt.title("Image 1 input")
img = pict[0].reshape(int(math.sqrt(pict.shape[1])),-1)
plt.imshow(img, cmap='gray')
plt.show()

plt.title("Image 1 recall")
img = pic1_recall.reshape(int(math.sqrt(pict.shape[1])),-1)
plt.imshow(img, cmap='gray')
plt.show()

plt.title("Image 2 input")
img = pict[1].reshape(int(math.sqrt(pict.shape[1])),-1)
plt.imshow(img, cmap='gray')
plt.show()

plt.title("Image 2 recall")
img = pic2_recall.reshape(int(math.sqrt(pict.shape[1])),-1)
plt.imshow(img, cmap='gray')
plt.show()

plt.title("Image 3 input")
img = pict[2].reshape(int(math.sqrt(pict.shape[1])),-1)
plt.imshow(img, cmap='gray')
plt.show()

plt.title("Image 3 recall")
img = pic3_recall.reshape(int(math.sqrt(pict.shape[1])),-1)
plt.imshow(img, cmap='gray')
plt.show()
        
#x2_recall = np.sign(np.dot(W,x2.T).T)
#x3_recall = np.sign(np.dot(W,x3.T).T)

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
#x1d = np.array([1., -1., 1., -1., 1., -1., -1., 1.]).reshape(1,8)
#x2d = np.array([1., 1., -1., -1., -1., 1., -1., -1.]).reshape(1,8)
#x3d = np.array([1., 1., 1., -1., 1., 1., -1., 1.]).reshape(1,8)
#
#for i in range(2):
#    x1d = np.sign(np.dot(W,x1d.T).T)
#    x2d = np.sign(np.dot(W,x2d.T).T)
#    x3d = np.sign(np.dot(W,x3d.T).T)
#    
## Plot recalls over inputs to ensure proper weight matrix configuration
#plt.title("Checking x1d convergence")
#plt.plot(x1[0], label='Attractor Input')
#plt.plot(x1d[0], label='Distorted Recall')
#plt.legend()
#plt.show()
#
#plt.title("Checking x2d convergence")
#plt.plot(x2[0], label='Attractor Input')
#plt.plot(x2d[0], label='Distorted Recall')
#plt.legend()
#plt.show()
#
#plt.title("Checking x3d convergence")
#plt.plot(x3[0], label='Attractor Input')
#plt.plot(x3d[0], label='Distorted Recall')
#plt.legend()
#plt.show()
