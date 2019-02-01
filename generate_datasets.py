import numpy as np
import os
import pickle
import pdb
import cv2
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', 
            default='/home/mike/DataSet/fewshot.dataset/')
    parser.add_argument('--dataset-name',
            default='tieredImagenet',
            help='tieredImagenet or miniImagenet or miniImagenet_cy')
    parser.add_argument('--output-path',
            default='./data_npy')
    args = parser.parse_args()
    return args

args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

if args.dataset_name == 'miniImagenet_cy':
    # downloaded from
    # https://github.com/cyvius96/prototypical-network-pytorch
    # download in data_root/dataset_name unzip images.zip
    imgs_root = os.path.join(args.data_root, args.dataset_name, 'images')
    csv_files = ['csv/{}.csv'.format(t) for t in ['train', 'val', 'test']]

    for dsettype in ['train', 'val', 'test']:
        csv_file = 'csv/{}.csv'.format(dsettype)
        print ('processing {}'.format(csv_file))
        lines = [x.strip() for x in open(csv_file, 'r').readlines()][1:]
        all_classes = []
        for line in lines:
            clsname = line.split(',')[-1]
            if not clsname in all_classes:
                all_classes.append(clsname)

        data_stacks = [[] for _ in range(len(all_classes))]
        for ii, line in enumerate(lines):
            imgpath, clsname = line.split(',')
            img = cv2.imread(os.path.join(imgs_root, imgpath))
            img = cv2.resize(img, (84,84), interpolation=cv2.INTER_LANCZOS4)
            data_stacks[all_classes.index(clsname)].append(img)

            if ii % 1000 == 0:
                print ('{:5d} / {:5d}'.format(ii, len(lines)))
        
        dataset_output_path = os.path.join(args.output_path, args.dataset_name)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        outfile_name = os.path.join(dataset_output_path, dsettype + '.npy')
        np.save(outfile_name, np.array(data_stacks))
        print ('saved in {}'.format(outfile_name))

 
if args.dataset_name == 'miniImagenet':
    # miniImagenet pkl file can be downloaded from
    # https://github.com/renmengye/few-shot-ssl-public
    # and locate it data_root/dataset_name and unzip it
    file_root = os.path.join(args.data_root, args.dataset_name)
    for dsettype in ['train', 'val', 'test']:
        fname = os.path.join(file_root, 'mini-imagenet-cache-{}.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        
        out_data = []
        for key, value in data['class_dict'].items():
            # key: classname
            # value : index of an image
            img = data['image_data'][value]
            out_data.append(img)

        dataset_output_path = os.path.join(args.output_path, args.dataset_name)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        outfile_name = os.path.join(dataset_output_path, dsettype + '.npy')
        np.save(outfile_name, np.array(out_data))
        print ('saved in {}'.format(outfile_name))


if args.dataset_name == 'tieredImagenet':
    # tieredImagenet pkl file can be downloaded from
    # https://github.com/renmengye/few-shot-ssl-public
    # and locate it data_root/dataset_name and unzip it
    file_root = os.path.join(args.data_root, args.dataset_name)
    for dsettype in ['train', 'val', 'test']:
        fname = os.path.join(file_root, '{}_images_png.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        images = np.zeros([len(data),84,84,3], dtype=np.uint8)
        for ii, item in tqdm(enumerate(data), desc='decompress'):
            img = cv2.imdecode(item, 1)
            images[ii] = img
        
        fname = os.path.join(file_root, '{}_labels.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            label = pickle.load(f, encoding='latin1')

        out_data = []
        labsp = label['label_specific']
        num_classes = np.unique(labsp)
        for i in num_classes:
            out_data.append(images[labsp==i])

        dataset_output_path = os.path.join(args.output_path, args.dataset_name)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        outfile_name = os.path.join(dataset_output_path, dsettype + '.npy')
        np.save(outfile_name, np.array(out_data))
        print ('saved in {}'.format(outfile_name))
