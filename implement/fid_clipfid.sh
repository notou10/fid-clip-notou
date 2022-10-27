#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python fid_no_patch.py \
--real_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt" \
--generated_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt_2"