import sys
import os
import numpy as np 

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

def read(path):
    fname_prefix = os.path.join(path, 'mini-imagenet-cache-')
    train_dict = unpickle(fname_prefix + 'train.pkl') 
    test_dict = unpickle(fname_prefix + 'test.pkl')
    val_dict = unpickle(fname_prefix + 'val.pkl')

    class_ids = list(train_dict['class_dict'].keys()) + \
            list(test_dict['class_dict'].keys()) + \
            list(val_dict['class_dict'].keys())
    d2ind = {}
    for i, c in enumerate(class_ids):
        d2ind[c] = i

    imgs_all = np.concatenate(\
            [train_dict['image_data'],
                test_dict['image_data'],
                val_dict['image_data']],
            axis=0)
    
    label1 = train_dict['class_dict']
    cd = 'class_dict'
    labels_all = []
    for ddict in [train_dict, test_dict, val_dict]:
        out_labels = np.zeros([len(ddict['image_data'])])
        for key, value in ddict['class_dict'].items():
            out_labels[value] = d2ind[key] 
        labels_all.append(out_labels)
    labels_all = np.concatenate(labels_all)

    print (imgs_all.shape)
    print (labels_all.shape)
    return imgs_all, labels_all


    






#
