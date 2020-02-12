# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:25:28 2020

@author: kwc57
"""

import math
import numpy as np 
import matplotlib.pyplot as plt


#used_data = data[:npatterns]


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
 

def weight_calc(patterns, do_scaling=True, disp_W=False, zeros_diagonal=True):
    # Check for 1-D pattern shape = (N,)
    if patterns.size == patterns.shape[0]:
        n_units = patterns.size
    else:
        n_units = patterns.shape[1]
    # This is the same that summing all the outer products of each pattern with itself
    W = np.dot(patterns.T,patterns)

    if do_scaling:
        W = W / n_units # Changed because /= was rasing an error for some reason...
    
    if zeros_diagonal:
        np.fill_diagonal(W,0)

    if disp_W:
        # Display weight matrix as greyscale image
        plt.title('Color representation of weight matrix')
        plt.imshow(W,  cmap='jet') 
        plt.colorbar()
        plt.show()
    return W


# Need to include number of rows, because we may ot want to store all images
def weight_calc_old(data, nrows, disp_W):
    # Calculate weight matrix
    W = np.dot(data[:nrows].T,data[:nrows])
    # Scale by number of points
    #W /= data.shape[1]
    if disp_W:
        # Display weight matrix as greyscale image
        plt.title('Greyscale representation of weight matrix')
        plt.imshow(W,  cmap='gray')
        plt.show()
    return W

"""
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
        image_vec_new = our_sign(np.dot(W,image_vec_prev.T).T)
        if (i+1)%print_step == 0:
                plt.title("update # %i" %(i+1))
                img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
                plt.imshow(img, cmap='gray')
                plt.show()
        fixed_point_found = check_fixed_point_found(image_vec_new, image_vec_prev)
        image_vec_prev = image_vec_new.copy()
    print("Number of epochs needed to find a fixed point " + str(i))
    return image_vec_new 
"""

def check_fixed_point_found(patterns_new, patterns_prev): 
    fixed_points_in_patterns = np.all(patterns_new == patterns_prev, axis = 1)
    fixed_point_found = np.any(fixed_points_in_patterns)
    return fixed_point_found


def degraded_recall_epochs_multiple_patterns(patterns_prev, W, epochs=1000, show_energy_per_epoch=False):
    n_patterns = patterns_prev.shape[0]
    n_nodes = patterns_prev.shape[1]
    patterns_new = np.zeros((n_patterns, n_nodes))
    for idx_pattern in range(n_patterns):   
        print("Index pattern: " + str(idx_pattern))
        this_pattern_prev = patterns_prev[idx_pattern]
        this_pattern_new = degraded_recall_epochs(this_pattern_prev, W, epochs, show_energy_per_epoch)
        patterns_new[idx_pattern] = this_pattern_new
    return patterns_new
    
    

def asynchronous_update(pattern_prev):
    n_nodes = len(pattern_prev)
    idx_nodes = np.array(range(n_nodes))
    np.random.shuffle(idx_nodes)
    for idx_node_i in idx_nodes:
        pass
    
    
    
    
    

    
def degraded_recall_epochs(pattern_prev, W, epochs=1000, show_energy_per_epoch=False):
    """
    pattern_prev: it's an array with one n_nodes values
    """
    n_nodes = len(pattern_prev)
    energy_per_epoch = []
    pattern_new = np.zeros(n_nodes)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        energy = calculate_energy(pattern_prev,W)
        print("Energy: " + str(energy))
        energy_per_epoch.append(energy)
        
        for idx_node_i in range(n_nodes):
            result_sum = 0
            for idx_node_j in range(n_nodes):
                result_sum += W[idx_node_i,idx_node_j] * pattern_prev[idx_node_j]
            pattern_new[idx_node_i] = our_sign(result_sum)    
            #patterns_new[idx_pattern, idx_node] = our_sign(patterns_prev[idx_pattern, :] @ W[idx_node])
        print("Pattern new:")
        print(pattern_new)
        
        # patterns_new = our_sign(np.dot(W,patterns_prev.T).T)
        
        if stability_reached(pattern_prev, pattern_new):
            print("Stability reached in " + str(epoch+1) + " epochs.")
            stability_epochs = epoch
            break
        
        pattern_prev = pattern_new.copy()  
    
    if show_energy_per_epoch:
        plt.plot(range(1, stability_epochs+2), energy_per_epoch, "c", linestyle='--', marker='o')
        plt.xlabel("Number of recall iterations")
        plt.ylabel("Energy")
        plt.show()
        print(energy_per_epoch)
    
    return pattern_new


def our_sign(x):
    if x>=0:
        return 1 
    else:
        return -1


def stability_reached(pattern_prev, pattern_new):
    return np.all(pattern_prev == pattern_new)
    


def degraded_recall(image_vec, W, epochs, print_step):
    # Plot input image
    plt.title("degraded input")
    img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
    plt.imshow(img, cmap='gray')
    plt.show()
    
    for i in range(epochs):
        image_vec = np.sign(np.dot(W,image_vec.T).T)
        if (i+1)%print_step == 0:
            plt.title("update # %i" %(i+1))
            img = image_vec.reshape(int(math.sqrt(image_vec.size)),-1)
            plt.imshow(img, cmap='gray')
            plt.show()  


def calculate_energy(pattern,W):
    return - pattern @ W @ pattern.T

