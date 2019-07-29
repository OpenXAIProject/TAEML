import numpy as np 
import os
import pickle
import cv2
from tqdm import tqdm
import pdb
from class_split import splits

def read_and_save(path):
    output_root = '../datasets'
    for dsettype in ['train', 'val', 'test']:
        fname = os.path.join(path, '{}_images_png.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        images = np.zeros([len(data),84,84,3], dtype=np.uint8)
        for ii, item in tqdm(enumerate(data), desc='decompress'):
            img = cv2.imdecode(item, 1)
            images[ii] = img

        fname = os.path.join(path, '{}_labels.pkl'.format(dsettype))
        with open(fname, 'rb') as f: 
            label = pickle.load(f, encoding='latin1')

        if dsettype=='train':
            # split the dataset by general label
            gn = label['label_general']
            sp = label['label_specific']
            gn_str = label['label_general_str']
            sp_str = label['label_specific_str']
            
            fine_label = sp
            coarse_label = sp.copy()
            
            # save full data
            out_data = []
            for i in range(len(sp_str)):
                out_data.append(images[fine_label==i])
            out_path = os.path.join(output_root, dsettype)
            out_text = os.path.join(out_path,
                    'tiered_full.npy')
            np.save(out_text, np.array(out_data))
            print ('saved in {}'.format(out_text))
            
            super_class = \
                    [0,0,0,0,
                    1,2,2,2,
                    2,3,3,3,
                    3,3,3,3,
                    3,3,3,3] 
            for i in range(len(coarse_label)):
                string = sp_str[sp[i]]
                coarse_label[i] = super_class[splits[string]]
                
            # save sub data 
            for i in np.unique(coarse_label):
                out_data = []
                current_labelset = fine_label[coarse_label==i]
                for jj in np.unique(current_labelset):
                    out_data.append(images[fine_label==jj])
                out_path = os.path.join(output_root, dsettype)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                out_text = os.path.join(out_path,
                        'tiered_sub{}.npy'.format(i))
                np.save(out_text, np.array(out_data))
                print ('saved in {}'.format(out_text))

        else:
            # just one specific set
            out_data = []
            labsp = label['label_specific']
            num_classes = np.unique(labsp)
            for i in num_classes:
                out_data.append(images[labsp==i])
            output_path = os.path.join(output_root, dsettype)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_file = os.path.join(output_path, 'tiered.npy')
            np.save(output_file, np.array(out_data))
            print ('saved in {}'.format(output_file))
