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
 

def noisy_pattern(pattern, noise_percent):
    N = pattern.size
    flat = np.copy(pattern).reshape(-1,N)
    ind = np.random.choice(N, int(N*noise_percent/100), replace=False)
    flat[0][ind] *= -1
    return flat.reshape(pattern.shape)
    



###################### 3.4 - Distortion Resistance ######################

#noise_percent = 50
#
## Create noisy pictures
#noisy1 = noisy_pattern(pict[0],noise_percent).reshape(1,-1)
#noisy2 = noisy_pattern(pict[1],noise_percent).reshape(1,-1)
#noisy3 = noisy_pattern(pict[2],noise_percent).reshape(1,-1)
## Recall original pictures from noisy pics
#noisy1_recall = hf.degraded_recall_epochs(noisy1, W, 100)
#noisy2_recall = hf.degraded_recall_epochs(noisy2, W, 100)
#noisy3_recall = hf.degraded_recall_epochs(noisy3, W, 100)
#
#
##Plot results - Image 1
#plt.title("Image 1 Original")
#plt.imshow(pict[0].reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#plt.title("Image 1 Noisy")
#plt.imshow(noisy1.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#plt.title("Image 1 Noisy Recall")
#plt.imshow(noisy1_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
##Plot results - Image 2
#plt.title("Image 2 Original")
#plt.imshow(pict[1].reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#plt.title("Image 2 Noisy")
#plt.imshow(noisy2.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#plt.title("Image 2 Noisy Recall")
#plt.imshow(noisy2_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
##Plot results - Image 3
#plt.title("Image 3 Original")
#plt.imshow(pict[2].reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#plt.title("Image 3 Noisy")
#plt.imshow(noisy3.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#plt.title("Image 3 Noisy Recall")
#plt.imshow(noisy3_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()


###################### 3.5 - Capacity ######################

####### Using images #######
pics_part = False
random_part = True

if pics_part == True:
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
    
    pict_for_learning=pict[:n_patterns]
    
    W = hf.weight_calc(pict_for_learning, disp_W = False, zeros_diagonal=False)
    pict_recall = hf.degraded_recall_epochs(pict_for_learning, W, epochs=1)
    #pict_recall2 = hf.vec_sign(np.dot(W,np.copy(pict_for_learning).T).T)
    
    plot_original_and_recall_imgs(pict_for_learning, pict_recall)
    #plot_original_and_recall_imgs(pict_for_learning, pict_recall2)
    
    stability_check = np.all(pict_recall==pict_for_learning)
    
    print("Are the patterns stable? " + str(stability_check))





####### Random pattern part #######
if random_part == True:
    n_patterns = 300
    n_units = 300
    #n_patterns = 3
    #n_units = 8
    
    bias = 0 # Skew towards -1 or 1
    noise_percent = 1
    
    rand_pat = hf.vec_sign(bias+np.random.randn(n_patterns,n_units))
    print("Average value: ", np.average(rand_pat))
    plt.imshow(rand_pat, cmap='gray')
    plt.show()
    
    #rand_pat = np.random.randint(0, 2, (n_patterns,n_units))
    rand_pat[rand_pat == 0] = -1
    
    noisy_pat = noisy_pattern(np.copy(rand_pat), noise_percent)    

    recall_count1 = np.zeros(n_patterns)
    recall_count2 = np.zeros(n_patterns)
    noise_recall_count1 = np.zeros(n_patterns)
    noise_recall_count2 = np.zeros(n_patterns)
    
    for i in range(n_patterns):
        #print(i) # Just to show speed / where you're at
        
        ### Recall without noise
        
        # Calculate weight matrix for all patterns up to & including i
        W1 = hf.weight_calc(np.copy(rand_pat[:i+1][:]),zeros_diagonal=False)
        W2 = hf.weight_calc(np.copy(rand_pat[:i+1][:]),zeros_diagonal=True)
        
        # Attempt recall of all patterns up to & including i
        recall1 = hf.vec_sign(np.dot(W1,np.copy(rand_pat[:i+1][:]).T).T)
        recall2 = hf.vec_sign(np.dot(W2,np.copy(rand_pat[:i+1][:]).T).T)
        
        #### I confirmed that dot product is identical (at least for 1 iteration)
        #recall1 = hf.degraded_recall_epochs(np.copy(rand_pat[:i+1][:]), W1, epochs=1)
        #recall2 = hf.degraded_recall_epochs(np.copy(rand_pat[:i+1][:]), W2, epochs=1)
    
        # Sum stability of all patterns up to & including i
        recall_count1[i] = np.sum(np.all(np.equal(recall1, rand_pat[:i+1][:]), axis=1))
        recall_count2[i] = np.sum(np.all(np.equal(recall2, rand_pat[:i+1][:]), axis=1))
        
        
        ### Recall with noise
        
        noise_recall1 = np.copy(noisy_pat[:i+1][:])
        noise_recall2 = np.copy(noisy_pat[:i+1][:])
        old1 = np.copy(noise_recall1)
        old2 = np.copy(noise_recall2) 
        
        for j in range(1000):
            noise_recall1 = hf.vec_sign(np.dot(W1,noise_recall1.T).T)
            noise_recall2 = hf.vec_sign(np.dot(W2,noise_recall2.T).T)
            
            if np.array_equal(noise_recall2, old2) or np.array_equal(noise_recall1, old1):
                print("converged: ", j)
                break
            
            old1 = np.copy(noise_recall1)
            old2 = np.copy(noise_recall2)
            

            
            
        
        noise_recall_count1[i] = np.sum(np.all(np.equal(noise_recall1, noisy_pat[:i+1][:]), axis=1))
        noise_recall_count2[i] = np.sum(np.all(np.equal(noise_recall2, noisy_pat[:i+1][:]), axis=1))
        
    plt.xlabel("Number of stored patterns")
    plt.plot(np.arange(n_patterns), recall_count1, "c", label="Stable pattern count | Weighted Diagonal")
    plt.plot(np.arange(n_patterns), recall_count2, "--k", label="Stable pattern count | Zero Diagonal")
    plt.legend()
    plt.show()    
    
    plt.xlabel("Number of recoverable noisy patterns (%i percent noise)" %noise_percent)
    plt.plot(np.arange(n_patterns), noise_recall_count1, "c", label="Stable pattern count | Weighted Diagonal")
    plt.plot(np.arange(n_patterns), noise_recall_count2, "--k", label="Stable pattern count | Zero Diagonal")
    plt.legend()
    plt.show()  

## Now do same with a splash of noise
















###################### Old Stuff ######################

####### Verifying noisy_pattern function  #######

#diff = np.sum(np.all(np.equal(rand_pat, noisy_pat), axis=1))
#diff = 100 - diff*100./n_patterns
#print("Percent of changed patterns: ", diff)
#
#plt.imshow(rand_pat, cmap='gray')
#plt.show()
#plt.imshow(noisy_pat, cmap='gray')
#plt.show()
    





#### Recall degraded patterns ###
#p10 = pict[9].reshape(1,-1)
##hf.degraded_recall(image_vec, W, epochs, print_step)
#print(" \n\n ############### p10 recall ############### ")
#p10_recall = hf.degraded_recall_epochs(p10, W, 1)
#plt.imshow(p10_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#
#p11 = pict[10].reshape(1,-1)
#print(" \n\n ############### p11 recall ############### ")
#p11_recall = hf.degraded_recall_epochs(p11, W, 4)
#plt.imshow(p11_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()
#    
#rand_vec = np.random.randint(2, size=pict.shape[1]).reshape(1,-1)
#rand_vec[rand_vec == 0] = -1
#print(" \n\n ############### random image recall ############### ")
#rand_recall = hf.degraded_recall_epochs(rand_vec, W, 100)
#plt.imshow(rand_recall.reshape(int(math.sqrt(pict.shape[1])),-1),  cmap='gray')
#plt.show()









