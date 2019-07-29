import os
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import voc2012
import mnist
import cifar10_cifar100
import caltech101_caltech256_cub200_2011_awa2
import omniglot
import miniImagenet
import tieredImagenet

PATH_RAW = '../raw/'
PATH_PKL = '../pkl/'

def read_datasets(str_dataset):
    print(str_dataset)
    if str_dataset == 'mnist':
        is_rgb = False

        labels_images_train = mnist.read('training', PATH_RAW + str_dataset)
        labels_images_test = mnist.read('testing', PATH_RAW + str_dataset)
        labels_train, images_train = zip(*labels_images_train)
        labels_test, images_test = zip(*labels_images_test)
        images_all = 255 - np.concatenate([images_train, images_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        print (images_all.shape)

    elif str_dataset == 'miniImagenet':
        is_rgb = True
        path_data = os.path.join(PATH_RAW, 'miniImagenet')
        images_all, labels_all = miniImagenet.read(path_data)

    elif str_dataset == 'tieredImagenet':
        is_rgb = True
        path_data = os.path.join(PATH_RAW, 'tieredImagenet')
        tieredImagenet.read_and_save(path_data)
        return

    elif str_dataset == 'cifar10':
        is_rgb = True

        path_data = os.path.join(PATH_RAW + str_dataset, 'cifar-10-batches-py')
        list_str_data_train = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5'
        ]
        str_data_test = 'test_batch'
        images_train = None
        labels_train = None

        for str_data_train in list_str_data_train:
            cur_images, cur_labels = cifar10_cifar100.load_data(path_data, str_data_train, True)
            if images_train is None:
                images_train = cur_images
            else:
                images_train = np.vstack((images_train, cur_images))

            if labels_train is None:
                labels_train = cur_labels
            else:
                labels_train = np.hstack((labels_train, cur_labels))

        images_test, labels_test = cifar10_cifar100.load_data(path_data, str_data_test, True)
        images_all = np.concatenate([images_train, images_test], axis=0) 
        labels_all = np.concatenate([labels_train, labels_test], axis=0)

    elif str_dataset == 'cifar100':
        is_rgb = True

        path_data = os.path.join(PATH_RAW + str_dataset, 'cifar-100-python')
        str_data_train = 'train'
        str_data_test = 'test'

        images_train, labels_train = cifar10_cifar100.load_data(path_data, str_data_train, False)
        images_test, labels_test = cifar10_cifar100.load_data(path_data, str_data_test, False)
        images_all = np.concatenate([images_train, images_test], axis=0) 
        labels_all = np.concatenate([labels_train, labels_test], axis=0)

    elif str_dataset == 'caltech101':
        is_rgb = True

        path_data = os.path.join(PATH_RAW + str_dataset, '101_ObjectCategories')
        images_all, labels_all = caltech101_caltech256_cub200_2011_awa2.read(path_data)

    elif str_dataset == 'caltech256':
        is_rgb = True

        path_data = os.path.join(PATH_RAW + str_dataset, '256_ObjectCategories')
        images_all, labels_all = caltech101_caltech256_cub200_2011_awa2.read(path_data)

    elif str_dataset == 'voc2012':
        is_rgb = True

        path_data = os.path.join(PATH_RAW + str_dataset, 'VOCdevkit/VOC2012/JPEGImages')
        path_classes = os.path.join(PATH_RAW + str_dataset, 'VOCdevkit/VOC2012/ImageSets/Main')
        images_train, labels_train, images_test, labels_test = voc2012.read(path_data, path_classes)
        images_all = np.concatenate([images_train, images_test], axis=0) 
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        
    elif str_dataset == 'cub200_2011':
        is_rgb = True

        path_data = os.path.join(PATH_RAW + str_dataset, 'CUB_200_2011/images')
        images_all, labels_all = caltech101_caltech256_cub200_2011_awa2.read(path_data)

    elif str_dataset == 'omniglot':
        is_rgb = False

        path_data = os.path.join(PATH_RAW + str_dataset, 'images_background')
        images_train, labels_train = omniglot.read(path_data)

        path_data = os.path.join(PATH_RAW + str_dataset, 'images_evaluation')
        images_test, labels_test = omniglot.read(path_data)
        images_all = np.concatenate([images_train, images_test], axis=0) 
        labels_all = np.concatenate([labels_train, labels_test], axis=0)

    elif str_dataset == 'awa2':
        is_rgb = True
        
        path_data = os.path.join(PATH_RAW + str_dataset, 'Animals_with_Attributes2/JPEGImages')
        images_all, labels_all = caltech101_caltech256_cub200_2011_awa2.read(path_data)


        
    else:
        raise ValueError('wrong str_dataset')

    dict_all = {
        'is_rgb': is_rgb,
        'dataset_name': str_dataset
    }
    images_all = np.array(images_all)
    labels_all = np.array(labels_all, dtype=np.uint16)


    print('images_all')
    print(images_all.shape)
    print(images_all.dtype)
    print('data shape: ', images_all[0].shape)
    print('min max : ', np.min(images_all[0]), np.max(images_all[0]))
    print('labels_all')
    print(labels_all.shape)
    print(labels_all.dtype)

    dict_all['data'] = images_all
    dict_all['labels'] = labels_all
    pickle.dump(dict_all, open('../pkl/{}.pkl'.format(str_dataset), 'wb'))

if __name__ == '__main__':
    if not os.path.isdir(PATH_PKL):
        os.makedirs(PATH_PKL)
#    read_datasets('mnist')
#    read_datasets('cifar10')
#    read_datasets('cifar100')
#    read_datasets('caltech101')
#    read_datasets('caltech256')
#    read_datasets('cub200_2011')
#    read_datasets('awa2')
#    read_datasets('omniglot')
#    read_datasets('voc2012') 
#    read_datasets('miniImagenet')
    read_datasets('tieredImagenet')
