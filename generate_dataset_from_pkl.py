import numpy as np
import os
import pickle
import pdb
import cv2
import argparse

# miniImagenet pkl file is downloaded from
# https://github.com/renmengye/few-shot-ssl-public

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', 
            default='/home/mike/DataSet/AAAI/aaai_raw/miniImagenet')
    args = parser.parse_args()
    return args

args = parse_args()
pkl_root = args.datadir
out_root = 'miniImagenet'

for dsettype in ['train', 'val', 'test']:
    fname = os.path.join(pkl_root, 'mini-imagenet-cache-{}.pkl').format(dsettype)
    with open(fname, 'rb') as f: 
        data = pickle.load(f)
    outpath = os.path.join(out_root, './{}').format(dsettype)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    out_data = []
    for key, value in data['class_dict'].items():
        # key : classname
        # value : indexes of img
        imgdata = data['image_data'][value]
        out_data.append(imgdata)
    np.save(os.path.join(outpath, 'miniImagenet.npy'), np.array(out_data))
