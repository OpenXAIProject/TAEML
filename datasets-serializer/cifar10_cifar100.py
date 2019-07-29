import sys
import numpy as np
import os
import cv2

if sys.version_info < (3,0): 
    # python 2
    import cPickle as pickle
    def unpickle(file_):
        with open(file_, 'rb') as fo:
            dict_ = pickle.load(fo)
        return dict_
else: 
    # python 3
    import pickle
    def unpickle(file_):
        with open(file_, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='latin1')
        return dict_

def load_data(path_data, str_data_train, is_cifar10, max_wh_size=84):
    dict_data = unpickle(os.path.join(path_data, str_data_train))
    cur_images = np.array(dict_data['data'])
    cur_images = np.reshape(cur_images, (cur_images.shape[0], 3, 32, 32))
    cur_images = np.transpose(cur_images, (0, 2, 3, 1))
    out_images = np.zeros([cur_images.shape[0],max_wh_size,max_wh_size,3], 
            dtype=np.uint8)
    for i in range(cur_images.shape[0]): 
        out_images[i] = cv2.resize(cur_images[i], (max_wh_size, max_wh_size))

    if is_cifar10:
        cur_labels = np.array(dict_data['labels'])
    else:
        cur_labels = np.array(dict_data['fine_labels'])
    return out_images, cur_labels
