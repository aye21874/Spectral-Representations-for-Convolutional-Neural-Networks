from .spectral_pooling import SpectralPooling
from .frequency_dropout import frequency_dropout

import tensorflow as tf
from tensorflow import keras

class spectralConv2D(keras.layers.Layer):
    
    def __init__(self, filters = 32,kernel_size=(3,3),strides=(1, 1), padding='VALID',activation=tf.nn.relu,input_shape=None):
        
        # Initialization
        super(spectralConv2D, self).__init__()
        assert len(kernel_size) > 1 , "Please input the Kernel Size as a 2D tuple"
        self.strides = strides 
        self.padding = padding
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        
    def build(self, input_shape):
        # Initialize Filters 
        # Assuming Input_shape is channels_last 
        kernel_shape = [self.kernel_size[0],self.kernel_size[1],input_shape[3],self.filters]
        
        self.kernel = self.add_weight( shape=(kernel_shape),
                                            name="Kernel_Real",
                                            initializer="glorot_uniform",
                                            trainable=True)
#         self.kernel_imag = self.add_weight( shape=kernel_shape,
#                                             name="Kernel_Imag",
#                                             initializer="glorot_uniform",
#                                             trainable=True)
        
        self.bias = self.add_weight(shape=(self.filters,),name="Bias", initializer="zeros", trainable=True)

    def call(self, inputs):
        
        complex_kernel = tf.cast(self.kernel,dtype=tf.complex64)
        
#         kernel = tf.complex(self.kernel_real,self.kernel_imag)
        
        fft2d_kernel = tf.signal.fft2d(complex_kernel)
        
        fft2d_kernel_real = tf.math.real(fft2d_kernel)
        fft2d_kernel_imag = tf.math.imag(fft2d_kernel)
        
        spectral_weight = tf.complex(fft2d_kernel_real,fft2d_kernel_imag)
        
        spatial_kernel = tf.signal.ifft2d(spectral_weight)
                
        spatial_kernel=tf.math.real(spatial_kernel)
        
        convolution_output = tf.nn.conv2d(
                            inputs,
                            spatial_kernel,
                            strides=[1, self.strides[0], self.strides[1], 1],
                            padding=self.padding
                        )

        convolution_output = tf.nn.bias_add(convolution_output, self.bias)
        
        if self.activation is not None:
            convolution_output= self.activation(convolution_output)
            
        return convolution_output


class Spectral_Pool(keras.layers.Layer):

    def __init__(
                    self,
                    layer_number,
                    total_layers,
                    gamma = 0.85,
                    frequency_dropout_alpha = 0.30,
                    frequency_dropout_beta = 0.15,
                    train_phase=False
                ):

        '''
        Following Keras Custom Layer Creation Steps : https://www.tensorflow.org/guide/keras/custom_layers_and_models

        Inputs to Init : 
                    self,
                    layer_number,
                    total_layers,
                    filter_shape=(3,3),
                    frequency_dropout_alpha = 0.30,
                    frequency_dropout_beta = 0.15,
                    train_phase=False
        
        Outputs : Nothing

        '''

        super(Spectral_Pool, self).__init__()
        

        self.layer_number = layer_number
        self.total_layers = total_layers
        self.gamma = gamma

        self.frequency_dropout_alpha = frequency_dropout_alpha
        self.frequency_dropout_beta = frequency_dropout_beta
        self.train_phase = train_phase


    def call(self,input_X):
        """
            Perform a single spectral pool operation.

        Args:
            input_x: Tensor representing a batch of channels-first images
                shape: (batch_size, num_channels, height, width)
            filter_size: int, the final dimension of the filter required
            freq_dropout_lower_bound: The lowest possible frequency
                above which all frequencies should be truncated
            freq_dropout_upper_bound: The highest possible frequency
                above which all frequencies should be truncated
            train_phase: tf.bool placeholder or Python boolean,
                but using a Python boolean is probably wrong

        Returns:
            An image of similar shape as input after reduction

        """
        assert len(input_X.shape) ==4 , "Input to Spectral_Pool has to be 4 Dimensions , BatchSize,Height,Width,Channels"

        filter_shape = [0,0]

        filter_shape[0] = int(self.gamma  * input_X.shape[1])
        filter_shape[1] = int(self.gamma  * input_X.shape[1])
        

        # Transpose Images to be channel first 

        images_channelfirst = tf.transpose(input_X,perm=[0,3,1,2])

        # Inputs will now have dimensions , Batchsize,Channels,Height,Width

        spectral_pool = SpectralPooling()

        # Get Fourier Image 
        fourier_images = spectral_pool.tf_fouriertransform(tf.cast(images_channelfirst, tf.complex64))

        # Apply low pass filter on Fourier Image 
        filtered_fourier_image = spectral_pool.tf_lowpassfilter(fourier_images,filter_shape)

        # Treat Corner Cases 
        filtered_fourier_image=spectral_pool.tf_treatcornercases(filtered_fourier_image)

        # Convert it back into spatial image 
        filtered_spatial_image= spectral_pool.tf_inversefouriertransform(filtered_fourier_image)

        # Do frequency Dropout 
        if self.train_phase:
            multiplier = self.frequency_dropout_alpha + (self.layer_number/self.total_layers)(self.frequency_dropout_beta - self.frequency_dropout_alpha)

            freq_dropout_lower_bound = tf.cast(multiplier*filtered_spatial_image[3],tf.int64)

            filtered_spatial_image = frequency_dropout(filtered_spatial_image,freq_dropout_lower_bound)

        # Keep only the real parts , and transpose it to  Batchsize,Height,Width,Channels
        filtered_spatial_image_channelslast = tf.math.real(tf.transpose(filtered_spatial_image,perm=[0,2,3,1]))

        return filtered_spatial_image_channelslast