import os
import numpy as np 
import cv2

def read(str_path, max_wh_size=84):
    if 'background' in str_path.split('/')[-1]:
        # one-shot train class set
        class_id = 0
    else:
        # one-shot test class set 
        class_id = 964
    imgs = []; labels = []
    for (path, dir, files) in os.walk(str_path):
        if 'character' in path: 
            for f in files:
                img = cv2.imread(os.path.join(path, f))
                img = cv2.resize(img,(max_wh_size, max_wh_size))
                imgs.append(img)
                labels.append(class_id)
            class_id += 1
            
    return np.array(imgs), np.array(labels)
