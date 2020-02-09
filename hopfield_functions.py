# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:25:28 2020

@author: kwc57
"""

import sys
import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random 

   
def weight_calc(data, nrows, disp_W):
    # Calculate weight matrix
    W = np.dot(data[:nrows].T,data[:nrows])
    # Scale by number of points
    W /= data.shape[1]
    if disp_W:
        # Display weight matrix as greyscale image
        plt.title('Greyscale representation of weight matrix')
        plt.imshow(W,  cmap='gray')
        plt.show()
    return W

def degraded_recall(image_vec, W, epochs, print_step):
    # Plot input image
    plt.title("degraded input")
    img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
    plt.imshow(img, cmap='gray')
    plt.show()
    
    for i in range(epochs):
        image_vec = np.sign(np.dot(W,image_vec.T).T)
        if (i+1)%print_step == 0:
            plt.title("update # %i" %(i+1))
            img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
            plt.imshow(img, cmap='gray')
            plt.show()  
