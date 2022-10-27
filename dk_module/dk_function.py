import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import PIL.Image
from PIL import ImageDraw
from PIL import ImageFont
import re
import requests
import time
import torch
from tqdm import tqdm
import torchvision
from torchvision import transforms
from typing import Any, \
                   Dict, \
                   List, \
                   Optional, \
                   Tuple
import yaml
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from dk_module.dk_dataloader import *
from PIL import Image
from itertools import combinations
import dnnlib
import torch.nn.functional as F
import pytorch_model_summary
#print(pytorch_model_summary.summary(inception_v3, torch.zeros(1,3,))

metrics_kwargs = dict(return_features=True)
feature_kwargs = dict(return_features=False)

device = torch.device('cuda:0')

_URL_TO_PKL_NAME = {'https://drive.google.com/uc?id=1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0': 'clip-vit_b32.pkl',
                    'https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC': 'ffhq-fid50k_5.30-snapshot-022608.pkl',
                    'https://drive.google.com/uc?id=1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL': 'inception_v3-tf-2015-12-05.pkl'}


def download_pickle(url: str,
                    pkl_name: str,
                    pickle_dir: str = './pickles/',
                    num_attempts: int = 10,
                    chunk_size: int = 512 * 1024,  # 512 KB.
                    retry_delay: int = 2) -> str:
    """Downloads network pickle file from an URL."""
    os.makedirs(pickle_dir, exist_ok=True)

    def _is_successful(response: requests.models.Response) -> bool:
        return response.status_code == 200

    # Download file from Google Drive URL.
    network_path = os.path.join(pickle_dir, pkl_name)
    if not os.path.exists(network_path):
        print(f'Downloading network pickle ({pkl_name})...')
        for attempts_left in reversed(range(num_attempts)):
            try:
                with requests.Session() as session:
                    with session.get(f'{url}&confirm=t', stream=True) as response:
                        assert _is_successful(response), \
                            f'Downloading network pickle ({pkl_name}) from URL {url} failed.'

                        # Save network pickle.
                        with open(network_path, 'wb') as f:
                            total = response.headers['Content-Length']
                            total = int(total)
                            pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                f.write(chunk)
                                pbar.update(len(chunk))
                        break
            except KeyboardInterrupt:
                raise
            except:
                print(f'Failed. Retrying in {retry_delay}s (attempts left {attempts_left})...')
                time.sleep(retry_delay)
    else:
        print(f'Downloading {pkl_name} skipped; already exists in {pickle_dir}')

    return network_path

def compute_fid(real_features: np.ndarray,
                gen_features: np.ndarray) -> float:
    """Computes the Fréchet Inception Distance."""
    assert real_features.ndim == 2 and gen_features.ndim == 2
    assert real_features.shape[0] == gen_features.shape[0]

    # Feature statistics.
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # FID.
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return fid

def load_feature_network(network_name: str) -> Any:
    """Loads a pre-trained feature network."""
    _network_urls = {#'clip': 'https://drive.google.com/uc?id=1VF0xYAfGEPH0bhNYLFS_yTEoVT2rkFFG',
                     'clip': 'https://drive.google.com/uc?id=1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0',
                     'inception_v3_tf': 'https://drive.google.com/uc?id=1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL'}
    assert network_name in _network_urls.keys(), \
        f"Unknown feature network name {network_name}."
    url = _network_urls[network_name]
    network_path = download_pickle(url=url,
                                   pkl_name=_URL_TO_PKL_NAME[url])

    with open(network_path, 'rb') as f:
        #model = torch.load(f)
        network = pickle.load(f)
        #import pdb; pdb.set_trace()
    #return model
    return network

def clip_transform():
    return transforms.Compose([
        #transforms.Resize((256,256)), 
        transforms.ToTensor(), #0~255 scale
        lambda x: (x*127.5+128).clamp(0, 255).to(torch.uint8)])

clip = load_feature_network(network_name='clip').to(device)
inception_v3 = load_feature_network(network_name='inception_v3_tf').to(device)


def compute(real_dir, generated_dir, total_img_num=10000, batch_size=200, H_W_same=True):
    
    #resolutions=256
    # if H_W_same==False:
    #     resolutions=None
        
    dataloader_imgnet_real = get_dataloader(zip_path=real_dir,
                                resolution=None, #None if your image's H, W is different
                                batch_size=batch_size,
                                num_images=real_dir)
    dataloader_imgnet_generated = get_dataloader(zip_path=generated_dir,
                                resolution=None, #None if your image's H, W is different
                                batch_size=batch_size,
                                num_images=real_dir)
    
    real_dataset_clip = torchvision.datasets.ImageFolder(root=f"{real_dir}", transform=clip_transform())
    generated_dataset_clip = torchvision.datasets.ImageFolder(root=f"{generated_dir}", transform=clip_transform())
    dataloader_clip_real = torch.utils.data.DataLoader(
                real_dataset_clip, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_clip_generated = torch.utils.data.DataLoader(
                generated_dataset_clip, batch_size=batch_size, shuffle=False, num_workers=4)

    real_metrics_features=[]
    fake_metrics_features=[]
    real_clip_features = []
    fake_clip_features = []
    
    #fid-imgnet
    #fid-imgnet real
    for num, (img, _) in tqdm(enumerate(dataloader_imgnet_real), total=len(dataloader_imgnet_real)):
        with torch.no_grad():
            img, _ = img.to(device), _.to(device)
            real_metrics_features.append(inception_v3(img, **metrics_kwargs).cpu().numpy())    
            if sum([a.shape[0] for a in real_metrics_features]) == total_img_num :break
    
    
    #fid-imgnet generated       
    for num, (img, _) in tqdm(enumerate(dataloader_imgnet_generated), total=len(dataloader_imgnet_generated)):
        with torch.no_grad():
            img, _ = img.to(device), _.to(device)
            fake_metrics_features.append(inception_v3(img, **metrics_kwargs).cpu().numpy())    
            if sum([a.shape[0] for a in fake_metrics_features]) == total_img_num :break
            
    
    #fid-clip  
    #fid-clip real
    for index, (data, _) in tqdm(enumerate(dataloader_clip_real), total=len(dataloader_clip_real)):
        with torch.no_grad():
            data,_= data.to(device, dtype=torch.uint8), _.to(device)
            real_clip_features.append(clip(data).cpu().numpy())
            if sum([a.shape[0] for a in real_clip_features]) == total_img_num :break
            
    #fid-clip generated
    for index, (data, _) in tqdm(enumerate(dataloader_clip_generated), total=len(dataloader_clip_generated)):
        with torch.no_grad():
            data,_= data.to(device, dtype=torch.uint8), _.to(device)

            fake_clip_features.append(clip(data).cpu().numpy())
            if sum([a.shape[0] for a in fake_clip_features]) == total_img_num :break
                
      
    real_metrics_features = np.concatenate(real_metrics_features, axis=0)
    fake_metrics_features = np.concatenate(fake_metrics_features, axis=0)
    real_clip_features = np.concatenate(real_clip_features, axis=0)   
    fake_clip_features = np.concatenate(fake_clip_features, axis=0)

    fid_imgnet = compute_fid(real_features=real_metrics_features,
                            gen_features=fake_metrics_features)

    fid_clip = compute_fid(real_features=real_clip_features,
                                gen_features=fake_clip_features)

    print("task: ",  real_dir.split("/")[-1], generated_dir.split("/")[-1])
    print(fid_imgnet, fid_clip)
    

        
    out_folder = f"./output/"    
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    
    f = open(f"{out_folder}/{real_dir.split('/')[-2]}.txt", 'a')
    f.write(f"task: {real_dir.split('/')[-1], generated_dir.split('/')[-1]} \n")
    f.write(f"fid-imgnet : {fid_imgnet} \n")
    f.write(f"fid-clip : {fid_clip} \n\n\n")
    f.close()
    