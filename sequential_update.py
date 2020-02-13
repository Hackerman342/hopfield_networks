
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: kwc57
"""

import math
import numpy as np 
import matplotlib.pyplot as plt
import hopfield_functions as hf
import img_functions



# Load data from pict
pict = np.genfromtxt('pict.dat', delimiter=',').reshape(-1,1024)

# Show images being used for training
show_images = False
if show_images:
    for i in range(pict.shape[0]):
        image = pict[i].reshape(int(math.sqrt(pict.shape[1])),-1)
        plt.imshow(image,  cmap='gray')
        plt.show()

# Calculate Weight Matrix
n_patterns = 3
disp_W = True

pict_for_learning=pict[:n_patterns]

W = hf.weight_calc(pict_for_learning, disp_W, zeros_diagonal=True)
pict_recall = hf.degraded_recall_epochs(pict_for_learning, W)

stability_check = np.all(pict_for_learning==pict_recall)

print("Are the patterns stable? " + str(stability_check))


img_functions.plot_original_and_recall_imgs(pict_for_learning, pict_recall)

######## SEQUENTIAL RECALL FOR DEGRADED PATTERNS 
print(" \n\n ############### p10 recall ############### ")
p10 = pict[9].reshape(1,-1)
img_functions.display_img(p10, "Image p10 input")

p10_recall = hf.degraded_recall_epochs(p10, W, show_energy_per_epoch=True)
img_functions.display_img(p10_recall, "Image p10 recall")


print(" \n\n ############### p11 recall ############### ")
p11 = pict[10].reshape(1,-1)
img_functions.display_img(p11, "Image p11 input")

p11_recall = hf.degraded_recall_epochs(p11, W, show_energy_per_epoch=True)
img_functions.display_img(p11_recall, "Image p11 recall")

"""
# Random image
rand_vec = np.random.randint(2, size=1024).reshape(1,-1)
rand_vec[rand_vec == 0] = -1
print(" \n\n ############### random image recall ############### ")
rand_recall = hf.degraded_recall_epochs(rand_vec, W, epochs=100)
plt.imshow(rand_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
plt.show()
"""

######## ASYNCHRONOUS RECALL FOR DEGRADED PATTERNS 

p10_recall_async = hf.degraded_recall_epochs(p10, W, type_of_update="async", show_energy_per_epoch=True)


p11_recall_async = hf.degraded_recall_epochs(p11, W, type_of_update="async", show_energy_per_epoch=True)



###################### 3.3 ######################

print("Energy at p0: " + str(hf.calculate_energy(pict_for_learning[0],W)))
print("Energy at p1: ", hf.calculate_energy(pict_for_learning[1],W))
print("Energy at p2: ", hf.calculate_energy(pict_for_learning[2],W))


print("Energy at p10: " + str(hf.calculate_energy(p10, W)))
print("Energy at p11: " + str(hf.calculate_energy(p11,W)))


########## NORMALLY DISTRIBUTED WEIGHT MATRIX ##########
n_nodes = pict.shape[1]
W_norm_dist = np.random.normal(size=(n_nodes,n_nodes))


########## NORMALLY SYMMETRIC DISTRIBUTED WEIGHT MATRIX ##########
W_symmetric = 0.5 * (W_norm_dist + W_norm_dist.T)
