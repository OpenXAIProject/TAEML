import os
import struct
import numpy as np
import cv2

def read(str_dataset, str_path, max_wh_size=84):
    if str_dataset == 'training':
        str_images = os.path.join(str_path, 'train-images-idx3-ubyte')
        str_labels = os.path.join(str_path, 'train-labels-idx1-ubyte')
    elif str_dataset == 'testing':
        str_images = os.path.join(str_path, 't10k-images-idx3-ubyte')
        str_labels = os.path.join(str_path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError('str_dataset')

    with open(str_labels, 'rb') as file_labels:
        magic, num = struct.unpack(">II", file_labels.read(8))
        labels = np.fromfile(file_labels, dtype=np.int8)

    with open(str_images, 'rb') as file_images:
        magic, num, rows, cols = struct.unpack(">IIII", file_images.read(16))
        images = np.fromfile(file_images, dtype=np.uint8).reshape(len(labels), rows, cols)

    get_images = lambda idx: (labels[idx], cv2.cvtColor(cv2.resize(images[idx],\
        (max_wh_size,max_wh_size)), cv2.COLOR_GRAY2RGB))

    for ind in range(0, len(labels)):
        yield get_images(ind)
