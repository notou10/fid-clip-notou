#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_resampling.py \
 --zip_path1=/mnt/hdd/dongkyun/PROJECTS/fid_clip/ffhq/temp/temp_50 \
 --zip_path2=/mnt/hdd/dongkyun/PROJECTS/fid_clip/ffhq/temp/temp_50 \
  --network_pkl=https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC \
  --feature_mode=pre_logits

  #--zip_path=data_path

# /home/dongkyun/hdd/PROJECTS/rosinality/ffhq/res256_generated_100
#  --zip_path1=/mnt/hdd/dongkyun/PROJECTS/fid_clip/ffhq/ffhq_gtgt \
#  --zip_path2=/mnt/hdd/dongkyun/PROJECTS/fid_clip/ffhq/stylegan_256_trunc1 \

