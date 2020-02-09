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

   
def weight_calc(data, nrows):
    # Calculate weight matrix
    W = np.dot(data[:nrows].T,data[:nrows])
    # Scale by number of points
    W /= data.shape[1]
    # Display weight matrix as greyscale image
    plt.title('Greyscale representation of weight matrix')
    plt.imshow(W,  cmap='gray')
    plt.show()
    return W