#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:09:21 2020

@author: flaviagv
"""

import matplotlib.pyplot as plt
import math 
import numpy as np 

def plot_original_and_recall_imgs(patterns, patterns_recall):
    for idx_pattern in range(len(patterns)):
        
        txt_title_orig = "Image " + str(idx_pattern+1) + " input"
        pattern = patterns[idx_pattern].reshape(1,-1)
        display_img(pattern,txt_title_orig)
        
        txt_title_recall = "Image " + str(idx_pattern+1) + " recall"
        pattern_recall = patterns_recall[idx_pattern].reshape(1,-1)
        display_img(pattern_recall, txt_title_recall)
       

def display_img(img, title_img):
    plt.title(title_img)
    plt.imshow(img.reshape(int(math.sqrt(img.shape[1])),-1),  cmap='gray')
    plt.show()
    
    
def generate_random_image(pixels_img=1024):
    rand_img = np.random.randint(2, size=pixels_img).reshape(1,-1)
    rand_img[rand_img == 0] = -1
    return rand_img