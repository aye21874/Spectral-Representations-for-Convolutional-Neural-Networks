#Taken from the ECBM4040 assignment #2

import numpy as np
from matplotlib import pyplot as plt
import os

try:
    from scipy.ndimage.interpolation import rotate
except ModuleNotFoundError:
    os.system('pip install scipy')
    from scipy.ndimage.interpolation import rotate

class ImageGenerator(object):
    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """        
        self.x = x
        self.y = y
        self.N = x.shape[0]
        self.is_bright = None
        self.is_horizontal_flip = None
        self.is_vertical_flip = None
        self.is_add_noise = None
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.bright = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N
    
    
    def create_aug_data(self):
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
        if self.bright:
            self.x_aug = np.vstack((self.x_aug,self.bright[0]))
            self.y_aug = np.hstack((self.y_aug,self.bright[1]))
            
        print("Size of training data:{}".format(self.N_aug))
        
    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """     
        samples = self.x_aug.shape[0]
        total_batches = samples // batch_size
        batch_count = 0
        x = self.x_aug
        y = self.y_aug
        while True:
            if (batch_count < total_batches):
                batch_count = batch_count + 1
                yield x[(batch_count-1)*batch_size:(batch_count)*batch_size,:,:,:],y[(batch_count-1)*batch_size:(batch_count)*batch_size]
            else:
                if shuffle:
                    shuffler = np.random.permutation(samples)
                    x = x[shuffler]
                    y = y[shuffler]
                batch_count = 0


    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        
        fig = plt.figure(figsize=(10, 10))

        for i in range(16):
            ax = fig.add_subplot(4, 4, i+1)
            ax.imshow(images[i, :].reshape(28, 28), 'gray')
            ax.axis('off')

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """

        translated = np.roll(self.x,shift_height,axis = 1)
        translated = np.roll(translated,shift_width,axis = 2)
        self.translated = (translated,self.y.copy())
        self.N_aug += self.N
        return translated


    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        """

        
        rotated = rotate(self.x,angle,axes = (1,2),reshape = False)
        self.rotated = (rotated,self.y.copy())
        self.N_aug += self.N
        return rotated

        

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        assert mode == 'h' or 'v' or 'hv'
        if mode == 'h':
            flipped = np.flip(self.x.copy(), axis=2)
            self.is_horizontal_flip = not self.is_horizontal_flip
        elif mode == 'v':
            flipped = np.flip(self.x.copy(), axis=1)
            self.is_vertical_flip = not self.is_vertical_flip
        elif mode == 'hv':
            flipped = np.flip(np.flip(self.x.copy(), axis=0), axis=1)
            self.is_horizontal_flip = not self.is_horizontal_flip
            self.is_vertical_flip = not self.is_vertical_flip
        else:
            raise ValueError('Mode should be \'h\' or \'v\' or \'hv\'')
        print('Vertical flip: ', self.is_vertical_flip, 'Horizontal flip: ', self.is_horizontal_flip)
    
        self.flipped = (flipped,self.y.copy())
        self.N_aug += self.N
        return flipped

    
    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        
        if not self.is_add_noise:
            self.is_add_noise = True
        noise = np.random.rand(self.x.shape[1],self.x.shape[2],self.x.shape[3])*amplitude
        samples = self.x.shape[0]
        added = self.x.copy()
        added  = added[0:int(samples*portion),:,:,:] + noise
        self.added = (added,self.y.copy())
        self.N_aug += self.N
        return added


    def brightness(self, factor):
        """
        Scale the pixel values to increase the brightness
        :param factor: A number greater than or equal to 1 that decides how each pixel in the image will be scaled. If factor is 2, then 
                       all pixel values will be doubled.
        :return bright: dataset with increased brightness
        """
        assert factor >= 1
        if not self.is_bright:
            self.is_bright = True
        bright = self.x.copy()
        for i in range(bright.shape[0]):
            bright[i, :, :, :] = (bright[i,:,:,:] * factor).astype(int)
            bright[i, :, :, :][bright[i,:,:,:] >= 255] = 255
            
        self.bright = (bright, self.y.copy())
        self.N_aug += self.N
        print("Brightness increased by a factor of:", factor)
        return bright

      