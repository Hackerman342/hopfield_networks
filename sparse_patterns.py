# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:11:54 2020

@author: matte
"""
import numpy as np
import random
import hopfield_functions as hf
import matplotlib.pyplot as plt
import tqdm


def generate_sparse_pattern(activity=0.5, dimensions=1024):
    img = np.zeros(dimensions)
    number_ones = int(activity*dimensions)
    indexes_ones = random.sample(range(dimensions), number_ones)
    img[indexes_ones] = 1
    return img

def distor_pattern(pattern, percentage):
    num_changed_cells = int(percentage/100 * pattern.shape[0])
    indexes = random.sample(range(pattern.shape[0]), num_changed_cells)
    new_pattern = pattern.copy()
    for idx in indexes:
        if new_pattern[idx] == 1:
            new_pattern[idx] = 0
        else:
            new_pattern[idx] = 1
    return new_pattern.reshape(1, -1)


n_imgs = 700
activity = 0.5
dimensions = 500

sparse_imgs = np.zeros((n_imgs, dimensions))

for idx_img in range(n_imgs):
    sparse_imgs[idx_img] = generate_sparse_pattern(activity=activity, dimensions=dimensions)

bias_values = np.arange(0, 25, 0.01)
capacity_per_bias = []
for bias in bias_values:
    print("Bias " + str(bias))
    storing_capability = 0
    for n_training_img in range(1, n_imgs):
        training_imgs = sparse_imgs[:n_training_img]
        #W = hf.weight_calc(patterns, zeros_diagonal=True, imbalanced=True, average_activity=activity)
        W = hf.weight_calc_sparse(training_imgs, activity)
        pict_recall = 0.5 + 0.5*hf.vec_sign(np.dot(W,np.copy(training_imgs).T).T - bias)
        if np.sum(np.all(np.equal(pict_recall, training_imgs), axis=1)) == n_training_img:
            storing_capability = n_training_img
        else:
            break
        #storing_capability = int(np.maximum(storing_capability, np.sum(np.all(np.equal(pict_recall, patterns), axis=1))))
        '''
        stability_check = np.all(training_imgs == pict_recall)
        if not stability_check:
            storing_capability = n_training_img - 1
            break
        '''
    capacity_per_bias.append(storing_capability)

plt.figure()
plt.plot(bias_values, capacity_per_bias)
plt.title("Dimensions = " + str(dimensions) + ", Activity = " + str(activity))
plt.xlabel("Bias")
plt.ylabel("Capacity")
plt.show()
#0.05 -> 11.79