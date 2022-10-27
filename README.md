# fid-clip

## with this repository, you can calculate fid-clip (encoder with pretrained CLIP viT), aligned with existing imagenet-fid. 


### Reproduction of FID-clip suggested in The Role of ImageNet Classes in Fréchet Inception Distance[https://arxiv.org/abs/2203.06026] (not official version)

Reproduction from(thanks to) https://github.com/kynkaat/role-of-imagenet-classes-in-fid 

due to 
1. pickle download issue,
2. want to calculate on generated sample not on-line generating

with existing repository, I repoduced fid-clip above.

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

then output will be saved like this 

<img width="380" alt="image" src="https://user-images.githubusercontent.com/45427036/198289542-7b08c74e-c477-41ad-aed0-0124bcf7fd8b.png">


---
## caution 

```
compute(real_dir, generated_dir, total_img_num=10000, batch_size = 200) 
```

in compute(), total_img_num%batch_size should be 0. if not, it will cause errors. 

somebody please fix it ㅋㅋ


