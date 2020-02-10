# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:25:28 2020

@author: kwc57
"""

import math
import numpy as np 
import matplotlib.pyplot as plt


#used_data = data[:npatterns]



def weight_calc(patterns, do_scaling=True, disp_W=False, zeros_diagonal=True):
    n_units = patterns.shape[1]
    
    # This is the same that summing all the outer products of each pattern with itself
    W = np.dot(patterns.T,patterns)
    
    if do_scaling:
        W /= n_units
    if zeros_diagonal:
        np.fill_diagonal(W,0)
    if disp_W:
        # Display weight matrix as greyscale image
        plt.title('Greyscale representation of weight matrix')
        plt.imshow(W,  cmap='jet') 
        plt.colorbar()
        plt.show()
    return W



def degraded_recall_first_to_converge(image_vec_prev, W, print_step=10, plot_patterns=False):
    # Plot input image
    if plot_patterns: 
        plt.title("degraded input")
        img = image_vec_prev.reshape(int(math.sqrt(image_vec_prev.size)),-1)
        plt.imshow(img, cmap='gray')
        plt.show()
    
    fixed_point_found=False
    i=0
    while ~fixed_point_found:
    #for i in range(epochs):
        i += 1
        image_vec_new = np.sign(np.dot(W,image_vec_prev.T).T)
        """if (i+1)%print_step == 0:
                plt.title("update # %i" %(i+1))
                img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
                plt.imshow(img, cmap='gray')
                plt.show()  """
        fixed_point_found = check_fixed_point_found(image_vec_new, image_vec_prev)
        image_vec_prev = image_vec_new.copy()
    print("Number of epochs needed to find a fixed point " + str(i))
    return image_vec_new 






def check_fixed_point_found(patterns_new, patterns_prev): 
    fixed_points_in_patterns = np.all(patterns_new == patterns_prev, axis = 1)
    fixed_point_found = np.any(fixed_points_in_patterns)
    return fixed_point_found





def degraded_recall_epochs(image_vec_prev, W, epochs=10):
    for i in range(epochs):
        image_vec_new = np.sign(np.dot(W,image_vec_prev.T).T)
        """if (i+1)%print_step == 0:
                plt.title("update # %i" %(i+1))
                img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
                plt.imshow(img, cmap='gray')
                plt.show()  """
        image_vec_prev = image_vec_new.copy()
    return image_vec_new 