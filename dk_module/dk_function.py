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

def fake_transform():
    return transforms.Compose([
        transforms.Resize((256,256)), 
        transforms.ToTensor(), #0~255 scale
        lambda x: (x*127.5+128).clamp(0, 255).to(torch.uint8)])

clip = load_feature_network(network_name='clip').to(device)
inception_v3 = load_feature_network(network_name='inception_v3_tf').to(device)


def compute(real_dir, generated_dir, total_img_num=10000, batch_size=200):
    dataloader_real = get_dataloader(zip_path=real_dir,
                                resolution=256,
                                batch_size=batch_size,
                                num_images=real_dir)
    fake_dataset = torchvision.datasets.ImageFolder(root=f"{generated_dir}", transform=fake_transform())
    fake_dataloader = torch.utils.data.DataLoader(
                fake_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    real_features = []
    real_metrics_features=[]
    fake_metrics_features=[]
    real_clip_features = []
    fake_features = []
    fake_clip_features = []
  
    for num, (img, _) in tqdm(enumerate(dataloader_real), total=len(dataloader_real)):
        with torch.no_grad():
            #img = torch.tensor(img, dtype = torch.uint8)
            img, _ = img.to(device), _.to(device)
            real_metrics_features.append(inception_v3(img.to(device), **metrics_kwargs).cpu().numpy())    
            real_clip_features.append(clip(img.to(device)).cpu().numpy())
            if sum([a.shape[0] for a in real_clip_features]) == total_img_num :break
      
    real_metrics_features = np.concatenate(real_metrics_features, axis=0)
    real_clip_features = np.concatenate(real_clip_features, axis=0)

    
    for index, (data, _) in tqdm(enumerate(fake_dataloader), total=len(fake_dataloader)):
        with torch.no_grad():
            data,_= data.to(device, dtype=torch.uint8), _.to(device)
            
            fake_features.append(inception_v3(data.to(device), **feature_kwargs).cpu().numpy())
            fake_metrics_features.append(inception_v3(data.to(device), **metrics_kwargs).cpu().numpy())    
            fake_clip_features.append(clip(data.to(device)).cpu().numpy())
            if sum([a.shape[0] for a in real_clip_features]) == total_img_num :break
    
    fake_metrics_features = np.concatenate(fake_metrics_features, axis=0)
    fake_clip_features = np.concatenate(fake_clip_features, axis=0)

    fid_imgnet = compute_fid(real_features=real_metrics_features,
                            gen_features=fake_metrics_features)

    fid_clip = compute_fid(real_features=real_clip_features,
                                gen_features=fake_clip_features)

    print("task: ",  real_dir.split("/")[-1], generated_dir.split("/")[-1])
    print(fid_imgnet, fid_clip)
    