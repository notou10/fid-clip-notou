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
from dk_module.dk_function import *
import glob
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('--real_dir', type=str)
parser.add_argument('--generated_dir', type=str)
args = parser.parse_args()


real_dir = args.real_dir
generated_dir = args.generated_dir

compute(real_dir, generated_dir, total_img_num=10000, batch_size = 200)

