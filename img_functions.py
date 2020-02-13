#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:09:21 2020

@author: flaviagv
"""

import matplotlib.pyplot as plt
import math 

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