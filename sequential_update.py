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


def plot_original_and_recall_imgs(patterns, patterns_recall):
    for idx_pattern in range(len(patterns)):
        txt_title_orig = "Image " + str(idx_pattern+1) + " input"
        plt.title(txt_title_orig)
        img = patterns[idx_pattern].reshape(int(math.sqrt(patterns.shape[1])),-1)
        plt.imshow(img, cmap='gray')
        plt.show()
        
        txt_title_recall = "Image " + str(idx_pattern+1) + " recall"
        plt.title(txt_title_recall)
        img = patterns_recall[idx_pattern].reshape(int(math.sqrt(patterns.shape[1])),-1)
        plt.imshow(img, cmap='gray')
        plt.show()
 


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

W = hf.weight_calc(pict_for_learning, disp_W)
pict_recall = hf.degraded_recall_epochs(pict_for_learning, W, epochs=1)

stability_check = np.all(pict_for_learning==pict_for_learning)

print("Are the patterns stable? " + str(stability_check))


energy_attractors = hf.calculate_energy(pict_for_learning,W)



plot_original_and_recall_imgs(pict_for_learning, pict_recall)

### Recall degraded patterns ###
p10 = pict[9].reshape(1,-1)

#hf.degraded_recall(image_vec, W, epochs, print_step)
print(" \n\n ############### p10 recall ############### ")
p10_recall = hf.degraded_recall_epochs(p10, W, 1)
plt.imshow(p10_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
plt.show()

p11 = pict[10].reshape(1,-1)
print(" \n\n ############### p11 recall ############### ")
p11_recall = hf.degraded_recall_epochs(p11, W, 4)
plt.imshow(p11_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
plt.show()
    
rand_vec = np.random.randint(2, size=1024).reshape(1,-1)
rand_vec[rand_vec == 0] = -1
print(" \n\n ############### random image recall ############### ")
rand_recall = hf.degraded_recall_epochs(rand_vec, W, 100)
plt.imshow(rand_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
plt.show()



###################### 3.3 ######################





