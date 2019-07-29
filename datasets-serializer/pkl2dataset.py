import numpy as np 
import os

try:
    import cPickle as pickle 
except:
    import pickle
import pdb

TRAIN_DATASET = ['awa2', 'cifar100', 'omniglot', 'voc2012', 'caltech256']
TEST_DATASET = ['mnist', 'cub200_2011', 'cifar10', 'caltech101', 'miniImagenet'] 
VALIDATION_DATASET = ['awa2', 'cifar100', 'omniglot', 'caltech256']

pkl_root = '../pkl'
dataset_list = os.listdir(pkl_root)
out_root = '../datasets'
if not os.path.exists(out_root):
    os.makedirs(out_root)

out_train = os.path.join(out_root, 'train')
out_test = os.path.join(out_root, 'test')
out_val = os.path.join(out_root, 'val')
if not os.path.exists(out_train):
    os.makedirs(out_train)
    os.makedirs(out_test)
    os.makedirs(out_val)


for pp, dsets in enumerate([TRAIN_DATASET, TEST_DATASET]):
    for datasetname in dsets:
        with open(os.path.join(pkl_root, datasetname+'.pkl'), 'rb') as f: 
            dataset = pickle.load(f) 
        name = dataset['dataset_name'] 
        print (name)

        # split by classes
        class_num = len(np.unique(dataset['labels']))
        class_data = [[] for i in range(class_num)]
        if dataset['data'][0].ndim == 3: 
            for i in range(len(dataset['data'])): 
                class_data[dataset['labels'][i]].append(dataset['data'][i])
        else:
            # it doesn't come here
            for i in range(len(dataset['data'])):
                rgb_data = np.expand_dims(dataset['data'][i], 2)
                rgb_data = np.tile(rgb_data, [1,1,3]) 
                class_data[dataset['labels'][i]].append(rgb_data)
        
        for i in range(class_num):
            class_data[i] = np.array(class_data[i]) 
        # save 
        if name in TEST_DATASET or name=='voc2012':
            if name in TEST_DATASET:
                out_name = os.path.join(out_test, name + '.npy')
            else:
                out_name = os.path.join(out_train, name + '.npy') 
            print ('full class: {}'.format(class_num))
            np.save(out_name, class_data)
        else:
            # train 80% val 20%
            out_name_train = os.path.join(out_train, name + '.npy')
            out_name_val = os.path.join(out_val, name + '.npy')
            split_class = int(class_num * 0.8)
            print ('full_class: {} / split_class: {}'.format(\
                    class_num, split_class))
            np.save(out_name_train, class_data[:split_class])
            np.save(out_name_val, class_data[split_class:])

# tiered imagenet is treated different from others
# it has own train/



#
#    elif imgnet_type=='tieredImagenet':
#        for dsettype in ['train', 'val', 'test']:
#            fname = os.path.join(path, '{}_images_png.pkl'.format(dsettype))
#            with open(fname, 'rb') as f:
#                data = pickle.load(f, encoding='bytes')
#            images = np.zeros([len(data),84,84,3], dtype=np.uint8)
#            for ii, item in tqdm(enumerate(data), desc='decompress'):
#                img = cv2.imdecode(item, 1) 
#                images[ii] = img
#
#            fname = os.path.join(path, '{}_labels.pkl'.format(dsettype))
#            with open(fname, 'rb') as f:
#                label = pickle.load(f, encoding='latin1')
#
#            out_data = []
#            labsp = label['label_specific']
#            num_classes = np.unique(labsp)
#            for i in num_classes:
#                out_data.append(images[labsp==i])
#
#            dataset_output_path = os.path.join(args.output_path, args.dataset_name)
#            if not os.path.exists(
