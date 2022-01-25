
# ECBM E4040 Neural Networks and Deep Learning
# This is a utility function to help you download the dataset and preprocess the data we use for this homework.
# requires several modules: _pickle, tarfile, glob. If you don't have them, search the web on how to install them.
# You are free to change the code as you like.

# Import modules. If you don't have them, try `pip install xx` or `conda
# install xx` in your console.

import _pickle as pickle
import os
import tarfile
import glob
import urllib.request as url
import numpy as np
import math


def max_pool(image, pool_size=2):

    imsize = image.shape[1]

    im_channel_last = np.moveaxis(image, 0, 2)
    
    im_new = im_channel_last.copy()
    
    for i in range(0, imsize, pool_size):
        for j in range(0, imsize, pool_size):
            max_val = np.max(im_channel_last[i: i + pool_size,
                                             j: j + pool_size],
                                             axis=(0, 1))
            im_new[i: i + pool_size, j: j + pool_size] = max_val
    im_new = np.moveaxis(im_new, 2, 0)
    return im_new



def approximation_loss(original_images, downsampled_images):

    number_ofimages = original_images.shape[0]
    
#     original_images = original_images.numpy()

    if not isinstance(downsampled_images, (np.ndarray) ):
        downsampled_images = downsampled_images.numpy()

    img = original_images.reshape(number_ofimages, -1)
    dimg = downsampled_images.reshape(number_ofimages, -1)

    # bring to same scale if not scales already
    if img.max() > 2:
        img = img / 255.
    if dimg.max() > 2:
        dimg = dimg / 255.

    error_norm = np.linalg.norm(img - dimg, axis=0)
    base_norm = np.linalg.norm(img, axis=0)
    
    return np.mean(error_norm / base_norm)


def download_data():
    """
    Download the CIFAR-10 data from the website, which is approximately 170MB.
    The data (a .tar.gz file) will be store in the ./data/ folder.
    :return: None
    """
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/cifar-10-python.tar.gz'):
        print('Start downloading data...')
        url.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                        "./data/cifar-10-python.tar.gz")
        print('Download complete.')
    else:
        print('CIFAR-10 package already exists.')

def load_data(mode='all'):
    """
    Unpack the CIFAR-10 dataset and load the datasets.
    :param mode: 'train', or 'test', or 'all'. Specify the training set or test set, or load all the data.
    :return: A tuple of data/labels, depending on the chosen mode. If 'train', return training data and labels;
    If 'test' ,return test data and labels; If 'all', return both training and test sets.
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/cifar-10-python.tar.gz'):
        download_data()
    else:
        print('./data/cifar-10-python.tar.gz already exists. Begin extracting...')
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists('./data/cifar-10-batches-py/'):
        package = tarfile.open('./data/cifar-10-python.tar.gz')
        package.extractall('./data')
        package.close()
    # Go to the location where the files are unpacked
    root_dir = os.getcwd()
    os.chdir('./data/cifar-10-batches-py')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    data_train = glob.glob('data_batch*')
    try:
        for name in data_train:
            handle = open(name, 'rb')
            cmap = pickle.load(handle, encoding='bytes')
            train_data.append(cmap[b'data'])
            train_label.append(cmap[b'labels'])
            handle.close()
        # Turn the dataset into numpy compatible arrays.
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label)
        handle = open('./test_batch', 'rb')
        cmap = pickle.load(handle, encoding='bytes')
        test_data.append(cmap[b'data'])
        test_label.append(cmap[b'labels'])
        test_data = np.array(test_data[0])
        test_label = np.array(test_label[0])
    except BaseException:
        print('Something went wrong...')
        return None
    os.chdir(root_dir)
    if mode == 'train':
        return train_data, train_label
    elif mode == 'test':
        return test_data, test_label
    elif mode == 'all':
        return train_data, train_label, test_data, test_label
    else:
        raise ValueError('Mode should be \'train\' or \'test\' or \'all\'')