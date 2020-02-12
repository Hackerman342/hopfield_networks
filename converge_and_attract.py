# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:31:13 2020

@author: kwc57
"""

import numpy as np 
import matplotlib.pyplot as plt
from itertools import permutations
import hopfield_functions as hf



def get_all_possible_patterns(): 
    a0 = [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,   1.]
    a1 = [-1.,  1.,  1.,  1.,  1.,  1.,  1.,   1.]
    a2 = [-1., -1.,  1.,  1.,  1.,  1.,  1.,   1.]
    a3 = [-1., -1., -1.,  1.,  1.,  1.,  1.,   1.]
    a4 = [-1., -1., -1., -1.,  1.,  1.,  1.,   1.]
    a5 = [-1., -1., -1., -1., -1.,  1.,  1.,   1.]
    a6 = [-1., -1., -1., -1., -1., -1.,  1.,   1.]
    a7 = [-1., -1., -1., -1., -1., -1., -1.,   1.]
    a8 = [-1., -1., -1., -1., -1., -1., -1.,   1.]
    a9 = [-1., -1., -1., -1., -1., -1., -1.,  -1.] 
    
    perm0 = set(permutations(a0))
    perm1 = set(permutations(a1))
    perm2 = set(permutations(a2))
    perm3 = set(permutations(a3))
    perm4 = set(permutations(a4))
    perm5 = set(permutations(a5))
    perm6 = set(permutations(a6))
    perm7 = set(permutations(a7))
    perm8 = set(permutations(a8))
    perm9 = set(permutations(a9))
    
    ar0 = np.array(list(perm0))
    ar1 = np.array(list(perm1))
    ar2 = np.array(list(perm2))
    ar3 = np.array(list(perm3))
    ar4 = np.array(list(perm4))
    ar5 = np.array(list(perm5))
    ar6 = np.array(list(perm6))
    ar7 = np.array(list(perm7))
    ar8 = np.array(list(perm8))
    ar9 = np.array(list(perm9))
    
    all_pos = np.concatenate((ar0, ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9), axis=0)
    
    all_pos_unique = np.unique(all_pos, axis=0)
    
    return all_pos_unique


if __name__ == "__main__":
    # Memory patterns
    x1 = np.array([-1., -1., 1., -1., 1., -1., -1., 1.]).reshape(1,-1)
    x2 = np.array([-1., -1., -1., -1., -1., 1., -1., -1.]).reshape(1,-1)
    x3 = np.array([-1., 1., 1., -1., -1., 1., -1., 1.]).reshape(1,-1)
    
    patterns = np.concatenate((x1,x2,x3), axis=0)
    
    # Calculate Weight Matrix (Scaling optional)
    #W = (1./8.)*(np.outer(x1,x1) + np.outer(x2,x2) + np.outer(x3,x3))
    W = hf.weight_calc(patterns, disp_W=True, zeros_diagonal=True,do_scaling=False)
     
    
    ### Check that x1, x2, x3 are fixed points
    ### Get xi based in precomputed W 
    """
    x1_recall = np.sign(np.dot(W,x1.T).T)
    x2_recall = np.sign(np.dot(W,x2.T).T)
    x3_recall = np.sign(np.dot(W,x3.T).T)
    """
    
    #patterns_recall = hf.degraded_recall_first_to_converge(patterns, W)
    
    
    
    #print("- Checking that x1, x2 and x3 are fixed points...")
    #print(np.all(patterns==patterns_recall))
    
    
    
    # Distorted memory patterns
    x1d = np.array([1., -1., 1., -1., 1., -1., -1., 1.]).reshape(1,-1) # 0 different
    x2d = np.array([1., 1., -1., -1., -1., 1., -1., -1.]).reshape(1,-1) # 0, 1 different
    x3d = np.array([1., 1., 1., -1., 1., 1., -1., 1.]).reshape(1,-1) # 0, 4 different
    
    patterns_distorted = np.concatenate((x1d,x2d,x3d), axis=0)
    
    
    
    patterns_distorted_recall = hf.degraded_recall_epochs_multiple_patterns(patterns_distorted, W)
    
    
    np.all(patterns_distorted_recall == patterns, axis=1)
    
    patterns_distorted_converg = np.all(patterns_distorted_recall == patterns, axis=1)
    
    print("Noisy patterns that converged: " )
    print(patterns_distorted_converg)
    
    
    
    # Plot recalls over distorted inputs to ensure proper 
    for idx_distorted_input in range(len(patterns_distorted_converg)):
        txt_title = "Checking x" + str((idx_distorted_input+1)) + "d convergence"
        plt.title(txt_title)
        plt.plot(patterns[idx_distorted_input], "c", label='Attractor Input')
        plt.plot(patterns_distorted_recall[idx_distorted_input],"--k", label='Distorted Recall')
        plt.legend()
        plt.show()
    
    
    all_possible_patterns = get_all_possible_patterns()
    
    
    all_possible_patterns_recall = hf.degraded_recall_epochs(all_possible_patterns, W, epochs=100)
    
        
    attractors = np.unique(all_possible_patterns_recall, axis=0)
    print("Number of attractors (assuming convergence): ", attractors.shape[0])
    
    
    all_possible_patterns_recall = hf.degraded_recall_epochs(all_possible_patterns, W, epochs=3)
    
        
    attractors = np.unique(all_possible_patterns_recall, axis=0)
    print("Number of attractors (assuming convergence): ", attractors.shape[0])
    
   
    
    ## Very distorted pattern 
    very_distorted_pattern = np.array([1., -1., -1., 1., -1., 1., -1., -1.]).reshape(1,-1) # 6 of the units of x1 were swipped
    
    very_distorted_pattern_recall = hf.degraded_recall_epochs(very_distorted_pattern, W, epochs=100)



