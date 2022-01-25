from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,ReLU,GlobalAveragePooling2D,Softmax# ,Dense, Flatten, , AveragePooling2D, MaxPooling2D,,
from tensorflow.keras import Model,Input

from .layers import Spectral_Pool,spectralConv2D


def build_model(input_shape,M=6, num_output=10,gamma=0.85,frequency_dropout_alpha=0.30,frequency_dropout_beta=0.15):

    model = Sequential()
        
    for i in range(1,M):
        cnn_kernel_size = (3,3)
        cnn_filters_count = 96 + (32*i)

        # First Start Off with a convolutional layer
        if i == 1 : 
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(Conv2D(cnn_filters_count, cnn_kernel_size,input_shape=input_shape, activation='relu'))

        else:
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(Conv2D(cnn_filters_count, cnn_kernel_size, activation='relu'))

        model.add(Spectral_Pool( layer_number = i , total_layers = M ,gamma = gamma,frequency_dropout_alpha = 0.30,frequency_dropout_beta = frequency_dropout_beta))

        model.add(ReLU())


    # Outside For Loop
    cnn_kernel_size = (1,1)
    cnn_filters_count = 96 + (32*M) 
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
    model.add(Conv2D(cnn_filters_count, cnn_kernel_size, activation='relu'))

    cnn_kernel_size = (1,1)
    cnn_filters_count = num_output
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
    model.add(Conv2D(cnn_filters_count, cnn_kernel_size, activation='relu'))


    model.add(GlobalAveragePooling2D())
    model.add(Softmax())


    return model

def build_model_5x5(input_shape,M=6, num_output=10,gamma=0.85,frequency_dropout_alpha=0.30,frequency_dropout_beta=0.15):

    model = Sequential()
        
    for i in range(1,M):
        cnn_kernel_size = (5,5)
        cnn_filters_count = 96 + (32*i)

        # First Start Off with a convolutional layer
        if i == 1 : 
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(Conv2D(cnn_filters_count, cnn_kernel_size,input_shape=input_shape, activation='relu'))

        else:
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(Conv2D(cnn_filters_count, cnn_kernel_size, activation='relu'))

        model.add(Spectral_Pool( layer_number = i , total_layers = M ,gamma = gamma,frequency_dropout_alpha = 0.30,frequency_dropout_beta = frequency_dropout_beta))

        model.add(ReLU())


    # Outside For Loop
    cnn_kernel_size = (1,1)
    cnn_filters_count = 96 + (32*M) 
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
    model.add(Conv2D(cnn_filters_count, cnn_kernel_size, activation='relu'))

    cnn_kernel_size = (1,1)
    cnn_filters_count = num_output
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
    model.add(Conv2D(cnn_filters_count, cnn_kernel_size, activation='relu'))


    model.add(GlobalAveragePooling2D())
    model.add(Softmax())


    return model


def build_spectral_conv_model_3x3(input_shape,M=6, num_output=10,gamma=0.85,frequency_dropout_alpha=0.30,frequency_dropout_beta=0.15):

    model = Sequential()
        
    for i in range(1,M):
        cnn_kernel_size = (3,3)
        cnn_filters_count = 96 + (32*i)

        # First Start Off with a convolutional layer
        if i == 1 : 
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size,input_shape=input_shape))

        else:
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size))

        model.add(Spectral_Pool( layer_number = i , total_layers = M ,gamma = gamma,frequency_dropout_alpha = 0.30,frequency_dropout_beta = frequency_dropout_beta))

        model.add(ReLU())


    # Outside For Loop
    cnn_kernel_size = (1,1)
    cnn_filters_count = 96 + (32*M) 
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
    model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size))


    cnn_kernel_size = (1,1)
    cnn_filters_count = num_output
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")

    model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size))


    model.add(GlobalAveragePooling2D())
    model.add(Softmax())


    return model


def build_spectral_conv_model_5x5(input_shape,M=6, num_output=10,gamma=0.85,frequency_dropout_alpha=0.30,frequency_dropout_beta=0.15):

    model = Sequential()
        
    for i in range(1,M):
        cnn_kernel_size = (5,5)
        cnn_filters_count = 96 + (32*i)

        # First Start Off with a convolutional layer
        if i == 1 : 
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size,input_shape=input_shape))

        else:
            print(f"Building Conv layer {i} with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
            model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size))

        model.add(Spectral_Pool( layer_number = i , total_layers = M ,gamma = gamma,frequency_dropout_alpha = 0.30,frequency_dropout_beta = frequency_dropout_beta))

        model.add(ReLU())


    # Outside For Loop
    cnn_kernel_size = (1,1)
    cnn_filters_count = 96 + (32*M) 
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")
    model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size))


    cnn_kernel_size = (1,1)
    cnn_filters_count = num_output
    print(f"Building Conv layer penultimate  with Filters count {cnn_filters_count} and cnn_kernel_size {cnn_kernel_size}")

    model.add(spectralConv2D(cnn_filters_count, cnn_kernel_size))


    model.add(GlobalAveragePooling2D())
    model.add(Softmax())


    return model