import os
import numpy as np
import cv2
import pdb

def center_crop(img):
    h,w,c = np.shape(img)  # (hwc)
    if h % 2 != 0: 
        return center_crop(img[:h-1,:,:]) 
    if w % 2 != 0:
        return center_crop(img[:,:w-1,:])
    if h > w: 
        out = img[h//2-w//2:h//2+w//2,:,:]
    elif h < w:
        out = img[:,w//2-h//2:w//2+h//2,:] 
    else:
        out = img
    return out

def read(path_data, path_classes, max_wh_size=84):
    list_dirs_data = os.listdir(path_data)
    list_dirs_classes = os.listdir(path_classes)
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []

    list_dirs_classes.sort()
    print(list_dirs_classes)
    print(len(list_dirs_data))

    idx_label = -1
    dict_str_data_all = {}
    for str_dir in list_dirs_classes:
        if '_train.' in str_dir:
            idx_label += 1
            str_target = 'train'
        elif '_val.' in str_dir:
            str_target = 'test'
        elif '_trainval.' in str_dir:
            continue
        else:
            if str_dir == 'train.txt' or str_dir == 'val.txt' or str_dir == 'trainval.txt':
                continue
            else:
                raise ValueError('not able to seperate')

        cur_dir = os.path.join(path_classes, str_dir)
        open_file = open(cur_dir, 'r')
        for elem_ in open_file:
            elem_ = elem_.split()
            cur_str_image = elem_[0] + '.jpg'
            if elem_[1] == '1':
                if cur_str_image in list_dirs_data:
                    if dict_str_data_all.get(cur_str_image) is not None:
                        dict_str_data_all[cur_str_image].append((str_target, idx_label))
                    else:
                        dict_str_data_all[cur_str_image] = list([(str_target, idx_label)])
                else:
                    raise ValueError('not existed')

    list_choices = []
    np.random.seed(42)
    shape_all = []
    for cur_str_image, val_cur in dict_str_data_all.items():
        if '.jpg' in cur_str_image:
            cur_image = cv2.imread(\
                    os.path.join(path_data, cur_str_image), cv2.IMREAD_COLOR)
            shape_all.append(cur_image.shape)
            cur_image = center_crop(cur_image)
            im_size_max = np.max(cur_image.shape[0:2])
            imscale = float(max_wh_size) / float(im_size_max)
            cur_image = cv2.resize(cur_image, None, None, fx=imscale,
                    fy=imscale, interpolation=cv2.INTER_LINEAR)

            cur_choice = np.random.choice(range(0, len(val_cur)))
            list_choices.append(cur_choice)
            cur_choice = val_cur[cur_choice]
            if cur_choice[0] == 'train':
                images_train.append(cur_image)
                labels_train.append(cur_choice[1])
            elif cur_choice[0] == 'test':
                images_test.append(cur_image)
                labels_test.append(cur_choice[1])
            else:
                raise ValueError('inappropriate type')
    print (np.mean(shape_all, axis=0))
    return images_train, labels_train, images_test, labels_test
