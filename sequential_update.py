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
        plt.title("Image # %i" %(i+1))
        plt.imshow(img,  cmap='gray')
        plt.show()

# Calculate Weight Matrix
n_patterns = 3
disp_W = True
W = hf.weight_calc_old(pict,n_patterns, disp_W) # Need to include # of rows in 
# Weight calcualtion, otherwise it tries to store all patterns (only want 
# first three here) and it forgets images (i.e.)

### Stability Check ####
stability_check = True
if stability_check:
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



### Recall degraded patterns ###
p10 = pict[9]

#hf.degraded_recall(image_vec, W, epochs, print_step)
print(" \n\n ############### p10 recall ############### ")
p10_recall = hf.degraded_recall(p10, W, 3, 1)

p11 = pict[10]
print(" \n\n ############### p11 recall ############### ")
p11_recall = hf.degraded_recall(p11, W, 3, 1)

    
rand_vec = np.random.randint(2, size=1024)
rand_vec[rand_vec == 0] = -1
print(" \n\n ############### random image recall ############### ")
rand_recall = hf.degraded_recall(rand_vec, W, 3, 1)
