#!/usr/bin/env bash

num_step=5
kshot=0

CUDA_VISIBLE_DEVICES=1 python main.py \
    --datasource=miniimagenet --metatrain_iterations=60000 \
    --meta_batch_size=1 --update_batch_size=5 \
    --num_updates=${num_step} --logdir=logs/tieredimgnet5way \
    --update_lr=9e-1 --resume=True --num_filters=32 --max_pool=True \
    --use_T=True --use_M=True --share_M=True --kshot=${kshot} \
    #--train=False

