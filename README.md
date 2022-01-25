# Spectral Representations for Convolutional Neural Networks

Implementation of Paper by Rippel et.al [Paper](https://arxiv.org/abs/1506.03767)

The project contains python notebooks which have the implmentation of the Spectral Pooling , frequency droput and the Spectral parameterization implementation proposed in the paper and efforts to replicate the results and key findings reported in the same.


## Authors
- [Vijay Kalmath](https://github.com/VijayKalmath)
- [Ayush Sinha](https://github.com/aye21874)
- [Arnax Saxena](https://github.com/DefinitelyArnav)


## Final Report

The Final Report is present as part of the report : [Final Report](E4040.2021Fall.SCNN.report.as6430.as6456.vsk2123.pdf) 

## Requirements

The project was developed using Tensorflow2.x and Keras. 

For the hyperparameter search we used Keras-Tuner which can installed by ```pip install keras-tuner --upgrade```

### Data 

The Data used is CIFAR10 and CIFAR100 hosted by the Computer Science department of [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

## Results and Trained Models 

The results and corresponding graphs are present in the report file. 

Preliminary results and layer testing jupyter notebooks are in the ```src/spectralParameterization-Notebooks``` and ```src/spectralPooling-Notebooks```. 

The jupyter notebooks on which the models were trained are in the ```src//final-JupyterNotebooks/``` folder.

The Trained Models are in the src/saved_models folder in h5 format apart from the customer layer CNNs which are stored in .pb format.

Initial Jupyter Notebooks used to understand and experiment with fourier jupyter notebooks are present in the ```playground``` folder.


## Code Organization

All code is located in the ```src``` folder. Within that folder, Python functions and classes that are shared between multiple notebooks are all located in the ```modules``` folder.

## Code Folder Structure 
```
./
├── LICENSE
├── README.md
├── References.txt
├── images
│   ├── Cameraman_Image.png
│   └── Male_Image.jpeg
├── playground
│   ├── FourierTransform+Lowpass_RGB-FrequencyDropout.ipynb
│   ├── FourierTransform+Lowpass_RGB.ipynb
│   ├── Fourier_Notebook.ipynb
│   ├── Fourier_lowpass.ipynb
│   ├── Images
│   │   └── Male_Image.jpeg
│   └── Spectral_Representation_ConvolutionalLayers.ipynb
└── src
    ├── __init__.py
    ├── final-JupyterNotebooks
    │   ├── spatial
    │   │   ├── deep
    │   │   │   ├── 3x3_Deep_SpatialCNN.ipynb
    │   │   │   ├── 5x5-Deep-SpatialCNN.ipynb
    │   │   │   ├── 5x5DeepSpatialCNN.h5
    │   │   │   └── history_5x5DeepSpatialCNN
    │   │   └── generic
    │   │       ├── 3x3-Generic-SpatialCNN.ipynb
    │   │       └── 5x5-Generic-SpatialCNN.ipynb
    │   ├── spectral
    │   │   ├── deep
    │   │   │   ├── 3x3-Deep-SpectralCNN.ipynb
    │   │   │   └── 5x5-Deep-SpectralCNN.ipynb
    │   │   └── generic
    │   │       ├── 3x3-Generic-SpectralCNN.ipynb
    │   │       └── 5x5-Generic-SpectralCNN.ipynb
    │   ├── spectral_convolution
    │   │   └── generic
    │   │       ├── 3x3-Generic-SpectralConvolutionCNN.ipynb
    │   │       └── 5x5-Generic-SpectralConvolutionCNN.ipynb
    │   └── spectral_pooling
    │       ├── 3x3-SpectralPoolingCNN-Convolution-CIFAR10.ipynb
    │       ├── 3x3-SpectralPoolingCNN-SpectralConvolution-CIFAR10.ipynb
    │       ├── 5x5-SpectralPoolingCNN-Convolution-CIFAR10.ipynb
    │       └── 5x5-SpectralPoolingCNN-SpectralConvolution-CIFAR10.ipynb
    ├── hyperparameter_search
    ├── modules
    │   ├── CNN_withSpectralPooling.py
    │   ├── Image_generator.py
    │   ├── frequency_dropout.py
    │   ├── layers.py
    │   ├── spectral_pooling.py
    │   └── utils.py
    ├── results
    │   ├── 3x3_deep_spatial
    │   ├── 3x3_deep_spectral
    │   ├── 3x3_generic_spatial
    │   ├── 3x3_generic_spectral
    │   ├── 3x3_spectral_convolution
    │   ├── 5x5_deep_spatial
    │   ├── 5x5_deep_spectral
    │   ├── 5x5_generic_spatial
    │   ├── 5x5_generic_spectral
    │   ├── 5x5_spectral_convolution
    │   ├── error_analysis.png
    │   ├── error_analysis_convolution.png
    │   └── optimization_convergence_analysis.ipynb
    ├── spectralParameterization-Notebooks
    │   ├── 2.1Spectral_Representation_ConvolutionLayer-test.ipynb
    │   ├── 2.2CNN-SpectralArchitecture-GenericModel-withoutDataAugmentation.ipynb
    │   └── 2.3CNN-SpectralArchitecture-DeepModel-withoutDataAugmentation.ipynb
    ├── spectralPooling-Notebooks
    │   ├── 1.1.SpectralPooling-Images-Figure2.ipynb
    │   ├── 1.2.SpectralPooling_ImagenetApproximationLosses.ipynb
    │   ├── 1.3.SpectralPoolingCNN-CIFAR10.ipynb
    │   ├── 1.4.SpectralPoolingCNN-CIFAR100.ipynb
    │   └── 1.5.BayesianHyperparameterSearch-SpectralCIFAR10-1.ipynb
    └── tensorflow
        └── spectral_pooling_cifar10
            ├── assets
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index

23 directories, 58 files
```
