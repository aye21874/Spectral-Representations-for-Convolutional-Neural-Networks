import numpy as np 
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt
import cv2
import math

def _frequency_drop_mask(fourierimage_size,freq_threshold):
                
        lowpass = np.ones(shape=fourierimage_size, dtype=np.float32)         
        
        if (fourierimage_size[3] + freq_threshold) % 2 == 1 : 
            freq_threshold += 1
            
        distance_from_corner = (fourierimage_size[3] - freq_threshold) // 2
        
        lowpass[:distance_from_corner,:] = 0
        lowpass[-distance_from_corner: ,:] = 0
        lowpass[:,:distance_from_corner] = 0
        lowpass[:,-distance_from_corner:] = 0
        
        
        return lowpass 

        

def frequency_dropout(fourier_images,freq_dropout_lower_bound):
    
    # Input fourier_images is Batchsize,Channels,Height,Width
    assert len(fourier_images.shape) == 4, "Input to frequency_dropout is not a four channel Array"
    
    freq_threshold = tf.random.uniform(shape=[], minval=freq_dropout_lower_bound, maxval=fourier_images.shape[2])

    lowpass = _frequency_drop_mask(fourier_images.shape,freq_threshold)
    
    filtered_image = fourier_image*lowpass

    return filtered_image
