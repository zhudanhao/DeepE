import torch
import numpy as np
import random
import math
from scipy.stats import rankdata
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_doubles(train,valid,test,words_indexes):
    train_keys = list(train.keys())
    train_doubles = []
    for h,r,t in train_keys:
        train_doubles.append((h,r,t))
        train_doubles.append((t,r+len(words_indexes),h))
    valid_keys = list(valid.keys())
    valid_doubles = []
    for h,r,t in valid_keys:
        valid_doubles.append((h,r,t))
        valid_doubles.append((t,r+len(words_indexes),h))
    test_keys = list(test.keys())
    test_doubles = []
    for h,r,t in test_keys:
        test_doubles.append((h,r,t))
        test_doubles.append((t,r+len(words_indexes),h))
    return train_doubles,valid_doubles,test_doubles

def get_rel_set(train,valid,test):
    rel_set = set()
    for h,r,t in train:
        rel_set.add(r)
    for h,r,t in valid:
        rel_set.add(r)
    for h,r,t in test:
        rel_set.add(r)
    return rel_set

def get_target_dict(train_doubles,x_valid,x_test):
    target_dict = {}
    for h,r,t in train_doubles:
        if (h,r) not in target_dict:
            target_dict[(h,r)] = set()
        target_dict[(h,r)].add(t)
    for h,r,t in x_valid:
        if (h,r) not in target_dict:
            target_dict[(h,r)] = set()
        target_dict[(h,r)].add(t)
    for h,r,t in x_test:
        if (h,r) not in target_dict:
            target_dict[(h,r)] = set()
        target_dict[(h,r)].add(t) 
    return target_dict   
    

head_scores_all = []
