def load_config(config_type):
    if config_type=='general':
        config = { 
            'TRAIN_DATASET': ['awa2', 'cifar100', 'omniglot', 'voc2012', 'caltech256'],
            'TEST_DATASET': ['mnist', 'cub200_2011', 'cifar10', 'caltech101', 'miniImagenet'],
            'VALIDATION_DATASET': ['awa2', 'cifar100', 'omniglot', 'caltech256']
            }
    elif config_type=='tiered':
        config = {
            'TRAIN_DATASET': ['tiered_sub{}'.format(i) for i in range(4)] + ['tiered_full'],
            'TEST_DATASET': ['tiered'],
            'VALIDATION_DATASET': ['tiered']
            }
    elif config_type=='tiered_test':
        config = {
            'TRAIN_DATASET': ['tiered_full'] \
                    + ['awa2', 'cifar100', 'caltech256'],
            'TEST_DATASET': ['tiered'],
            'VALIDATION_DATASET': ['tiered', 'awa2', 'cifar100', 'caltech256'],
            }
    return config
