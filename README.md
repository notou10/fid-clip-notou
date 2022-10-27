# fid-clip

Reproduction from(thanks to) https://github.com/kynkaat/role-of-imagenet-classes-in-fid 
(due to 1. pickle download issue, 2. want to calculate on generated sample not on-line generating)

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

in notod_fid folder, 

`
bash implement/fid_clipfid 0
`


(0 is allocated gpu num. which you can change if you want)

---

in fid_clipfid.sh, you should place your real/generated images folder like this below


`
CUDA_VISIBLE_DEVICES=$1 python fid_no_patch.py \
--real_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt" \
--generated_dir "/mnt/hdd/dongkyun/PROJECTS/fid_clip/salient/lsun_cat/gt_2"
`

