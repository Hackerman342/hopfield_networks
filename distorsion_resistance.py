# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: kwc57
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: kwc57
"""

import math
import numpy as np 
import matplotlib.pyplot as plt
import hopfield_functions as hf
import random

def distor_pattern(pattern, percentage):
    num_changed_cells = int(percentage/100 * pattern.shape[0])
    indexes = random.sample(range(pattern.shape[0]), num_changed_cells)
    new_pattern = pattern.copy()
    for idx in indexes:
        if new_pattern[idx] == 1:
            new_pattern[idx] = -1
        else:
            new_pattern[idx] = 1
    return new_pattern


# Load data from pict
pict = np.genfromtxt('pict.dat', delimiter=',').reshape(-1,1024)

# Show images being used for training
show_images = False
if show_images:
    for i in range(pict.shape[0]):
        img = pict[i].reshape(int(math.sqrt(pict.shape[1])),-1)
        plt.imshow(img,  cmap='gray')
        plt.show()

# Calculate Weight Matrix
n_patterns = 3
disp_W = True


pict_for_learning=pict[:n_patterns]

for percentage in range(101):
    W = hf.weight_calc(pict_for_learning, disp_W, zeros_diagonal=True)
    
    pict_recall = hf.degraded_recall_epochs_multiple_patterns(pict_for_learning, W)
    distorted_pattern = distor_pattern(pict_for_learning[0], percentage)
    stability_check = np.all(pict_for_learning[0] == distorted_pattern)
    
    print("Percentage " + str(percentage) + "; Stability" + str(stability_check))








