#TRAIN_DATASETS = ['awa2', 'cifar100', 'omniglot', 'voc2012', 'caltech256']
TRAIN_DATASETS = ['cifar100']
TEST_DATASETS = ['mnist', 'cub200_2011', 'cifar10', 'caltech101']

TRAIN_DATASETS_SIZE = {
    'awa2'          : int(37322*0.8),
    'omniglot'      : int(32460*0.8),
    'caltech256'    : int(30607*0.8),
    'cifar100'      : int(60000*0.8),
    'voc2012'       : 11540
}

ALL_DATASETS_SIZE = {
    'awa2'          : int(37322*0.8),
    'omniglot'      : int(32460*0.8),
    'caltech256'    : int(30607*0.8),
    'cifar100'      : int(60000*0.8),
    'miniImagenet'  : int(60000*0.64),
    'cifar10'       : 60000,
    'voc2012'       : 11540,
    'caltech101'    : 9144,
    'mnist'         : 70000,
    'cub200_2011'   : 11788
}

TEST_NUM_EPISODES = 600
