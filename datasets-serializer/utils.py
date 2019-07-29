import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import sklearn.model_selection

def unpickle(file_):
    with open(file_, 'rb') as fo:
        dict_ = pickle.load(fo)
    return dict_

def split_datasets(X, y, is_test, keep_ratio, test_size=0.1, valid_size=0.1, random_state=42):
    if is_test:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None
    
    X_train_orig, X_valid, y_train_orig, y_valid = sklearn.model_selection.train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state, stratify=y_train)
    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train_orig, y_train_orig, train_size=keep_ratio, random_state=random_state, stratify=y_train_orig)
    print(X_train_orig.shape)
    print(y_train_orig.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    if is_test:
        print(X_test.shape)
        print(y_test.shape)
    return X_train_orig, X_train, X_valid, X_test, y_train_orig, y_train, y_valid, y_test

