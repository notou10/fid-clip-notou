#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python fid_no_patch.py \
--real_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_fused/set2/house" \
--generated_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_fused/set2/ground" \
--batch_size 200 \
--total_img_num 10000


#--generated_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt" \
#"dongkyun/PROJECTS/fid_clip/salient/lsun_fused/set1/ground/" "dongkyun/PROJECTS/fid_clip/salient/lsun_fused/set1/sky/"
#python -m pytorch_fid dongkyun/PROJECTS/fid_clip/salient/lsun_fused/set2/house/a dongkyun/PROJECTS/fid_clip/salient/lsun_fused/set2/ground/a