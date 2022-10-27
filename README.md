# fid-clip

## with this repository, you can calculate fid-clip (encoder with pretrained CLIP viT), with existing imagenet-fid. 

Reproduction from(thanks to) https://github.com/kynkaat/role-of-imagenet-classes-in-fid 

(due to 
1. pickle download issue,
2. want to calculate on generated sample not on-line generating
with existing repository, I repoduced this repository)

# prerequisites 

should create environment first

`
conda env create -f environment.yml
`

`
conda activate imagenet-classes-in-fid
`

---
# usage

in notou_fid folder, 

`
bash implement/fid_clipfid 0
`


(0 is allocated gpu num. which you can change if you want)

---

in fid_clipfid.sh, you should place your real/generated images folder like this below


```
CUDA_VISIBLE_DEVICES=$1 python fid_no_patch.py \
--real_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt" \
--generated_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt_2"
--batch_size 200 \
--total_img_num 10000
```

total_img_num : you can calculate only subset of your datasets for speed.

