import os
import numpy as np
import cv2

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

def read(path_data, max_wh_size=84):
    list_dirs = os.listdir(path_data)
    images_all = []
    labels_all = []
    names_all = [] 

    idx_label = 0

    for str_dir in list_dirs:
        cur_dir = os.path.join(path_data, str_dir)
        list_cur_files = os.listdir(cur_dir)
        cur_count = 0 
        for str_cur_file in list_cur_files:
            if '.jpg' in  str_cur_file:
                cur_image = cv2.imread(os.path.join(cur_dir, str_cur_file))
                cur_image = center_crop(cur_image)
                im_size_max = np.max(cur_image.shape[0:2])
                im_scale = float(max_wh_size) / float(im_size_max)
                cur_image = cv2.resize(cur_image, None, None, fx=im_scale,
                        fy=im_scale, interpolation=cv2.INTER_AREA)
                
                images_all.append(cur_image)
                labels_all.append(idx_label)
                names_all.append(str_cur_file)
                cur_count += 1
            else:
                print(os.path.join(cur_dir, str_cur_file))
        idx_label += 1
    return images_all, labels_all
