import numpy as np
import os
import time
import sys
import pickle
from lib.params import TRAIN_DATASETS, TEST_DATASETS

class EpisodeGenerator():
    def __init__(self, data_dir, phase):
        if phase == 'train':
            self.dataset_list = TRAIN_DATASETS
        elif phase == 'test':
            self.dataset_list = TEST_DATASETS
        else:
            raise ValueError('train or test phases')
        self.data_root = data_dir
        self.dataset = {}
        self.data_all = []
        self.y_all = []
        for i, dname in enumerate(self.dataset_list):
            load_dir = os.path.join(data_dir,
                    phase, dname+'.npy')
            dataset = np.load(load_dir)
            dataset_ = []
            for j in range(dataset.shape[0]):
                data_ = dataset[j]
                #data_ = np.transpose(data_, (0, 3, 1, 2))
                dataset_.append(data_)
            dataset_ = np.asarray(dataset_)
            self.dataset[dname] = dataset_

    def get_episode(self, nway, kshot, qsize, dataset_name=None,
                    onehot=True, normalize=False, if_singleq=False):

        if not dataset_name:
            dataset_name = self.dataset_list[np.random.randint(len(self.dataset_list))]
        dd = self.dataset[dataset_name]
        random_class = np.random.choice(len(dd), size=nway, replace=False)
        support_set_data = []; query_set_data = []
        support_set_label = []; query_set_label = []

        for n, rnd_c in enumerate(random_class):
            data = dd[rnd_c]
            rnd_ind = np.random.choice(len(data), size=kshot+qsize, replace=False)
            rnd_data = data[rnd_ind]

            label = np.array([n for _ in range(kshot+qsize)])
            support_set_data += [r for r in rnd_data[:kshot]]
            support_set_label += [l for l in label[:kshot]]

            query_set_data += [r for r in rnd_data[kshot:]]
            query_set_label += [l for l in label[kshot:]]

        support_set_data = np.reshape(support_set_data,
                [-1] + list(rnd_data.shape[1:]))
        query_set_data = np.reshape(query_set_data,
                [-1] + list(rnd_data.shape[1:]))

        if normalize:
            support_set_data = support_set_data.astype(np.float32) / 255.
            query_set_data = query_set_data.astype(np.float32) / 255.

        if onehot:
            s_1hot = np.zeros([nway*kshot, nway])
            s_1hot[np.arange(nway*kshot), support_set_label] = 1
            q_1hot = np.zeros([nway*qsize, nway])
            q_1hot[np.arange(nway*qsize), query_set_label] = 1
            support_set_label = s_1hot
            query_set_label = q_1hot

        if if_singleq:
            single_ind = np.random.randint(len(query_set_data))
            query_set_data = query_set_data[np.newaxis,single_ind]
            query_set_label = query_set_label[np.newaxis,single_ind]

        return support_set_data, support_set_label, query_set_data, query_set_label
