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
import img_functions as imgf
import time

def distor_pattern(pattern, percentage):
    num_changed_cells = int(percentage/100 * pattern.shape[0])
    indexes = random.sample(range(pattern.shape[0]), num_changed_cells)
    new_pattern = pattern.copy()
    for idx in indexes:
        if new_pattern[idx] == 1:
            new_pattern[idx] = -1
        else:
            new_pattern[idx] = 1
    return new_pattern.reshape(1, -1)


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
selected_pict = pict_for_learning[0]
percentages = np.arange(0, 101, 10)
accuracy = []
W = hf.weight_calc(pict_for_learning, disp_W, zeros_diagonal=True)

for percentage in percentages:
    distorted_pattern = distor_pattern(selected_pict, percentage)
    restored_pics = 0
    for i in range(100):
        pict_recall = hf.degraded_recall_epochs(distorted_pattern, W, epochs = 1000).reshape(1, -1)
        stability_check = np.all(selected_pict == pict_recall)
        print("Percentage " + str(percentage) + "; Stability" + str(stability_check))
        #imgf.plot_original_and_recall_imgs(distorted_pattern, pict_recall)
        if stability_check:
            restored_pics += 1
    accuracy.append(restored_pics/100)

plt.figure()
plt.plot(percentages, accuracy)
plt.xlabel("Percentage noise")
plt.ylabel("Accuracy")
plt.title("Noise accuracy over 100 iterations")






