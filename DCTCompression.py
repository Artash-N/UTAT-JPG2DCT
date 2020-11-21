# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:13:41 2020

@author: Artash Nath

Co-Founder, HotPopRobot.com
Twitter: @wonrobot
"""

# Importing Required Libararies
from scipy.fftpack import dct, idct # To perform DCT And Inverse DCT
from skimage.io import imread # Read Image File
import numpy as np # Numpy to perform mathematical operations on images/arrays
import matplotlib.pylab as plt # Plot results and display images



# Calculates multidimension Discrete Cosine Transformation on an image
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho') #Converts an array into an array of DCT Coefficients of same shape

# Calculates Inverse Diicrete Cosine Transformation on an image
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho') # Reverses an array of DCT Coefficients back into an image

# Calculate (%) simmilarity between 2 arrays of same size

def image_simmilarity(imageA, imageB):
    
    diff = abs(imageA - imageB) # Find difference between 2 images
    
    err = diff.mean() # Calculate Average Difference between those 2 images
    
    return err


# Function that takes an image, converts it into DCT Coefficients. Keeping only percentage "thresh" of coefficients
# And reversing kept coefficients back into image

def DCTrecreate(im, thresh, display = False):
    
    DCT =  dct2(im) # Calculates and Creates an array of DCT Coefficients equal to the size of the original image
    thresh2 = np.sort(abs((DCT.ravel())))[int(DCT.size*(1-thresh))] # Determines lowest DCT Coeff values to keep based on "thresh"
    dct_thresh = DCT * (abs(DCT) > (thresh2)) # Removes all DCT Vales lower then determines threshold from DCT Coeff. Array
    P = round((np.sum(abs(DCT) > (thresh2))/im.size)*100, 2) # Re-Calculates threshold of pixels discarded for verification
    r_im = idct2(dct_thresh) # Reverses array of kept DCT Coeffs. Back into a reconstructed image
    r_im =r_im.astype(np.uint8)
    simm = image_simmilarity(im, r_im)
    #Display Orginal VS Recreated Image if Display==True
    if display == True:
        fig, axs = plt.subplots(1, 2, figsize = (16,8))
        axs[0].imshow(im)
        axs[0].axis('on')
        axs[0].set_title("Original Image")

        axs[1].imshow(r_im)
        axs[1].axis('off')
        title = "Reconsturcted Image ("+str(P)+"% Coefficients) "+ "| Simmilarity : "+str(int(simm))
        axs[1].set_title(title)
        plt.show()
        
    #Return recreated image array
    
    
    else:
        return r_im, simm
    
    
def DCTEncode(im, thresh):
    DCT =  dct2(im) # Calculates and Creates an array of DCT Coefficients equal to the size of the original image
    thresh2 = np.sort(abs((DCT.ravel())))[int(DCT.size*(1-thresh))] # Determines lowest DCT Coeff values to keep based on "thresh"
    dct_thresh = DCT * (abs(DCT) > (thresh2)) # Removes all DCT Vales lower then determines threshold from DCT Coeff. Array
    #P = round((np.sum(abs(DCT) > (thresh2))/im.size)*100, 2) # Re-Calculates threshold of pixels discarded for verification
    
    return dct_thresh

def DCTDecode(dct, return_simmilarity=False, original_image = None):
    if (not return_simmilarity) and original_image==None:
        raise RuntimeError("original_image must be defined to return simmilarity between original and recreated image")
    elif type(dct)!=np.ndarray:
        raise ValueError("dct must be a numpy array")
    elif type(dct)!=np.ndarray:
        raise ValueError("dct must be a numpy array")
        
    else:
        
        r_im = idct2(dct) # Reverses array of kept DCT Coeffs. Back into a reconstructed image
        r_im =r_im.astype(np.uint8)
        
    
        if return_simmilarity:
            simm = image_simmilarity(original_image, r_im)
            return r_im, simm
        else:
            return r_im
