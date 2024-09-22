import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils import to_psnr, print_log, validation, adjust_learning_rate
import os
import numpy as np
import random
from models.restorer import Restorer

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-val_root', help='test dataset path', default='', type=str)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=18, type=int)
parser.add_argument('--task', help='degradation type', default='low light', choices='[low light,snow,haze,blur,noise,rain]')
parser.add_argument('--ckpt_path', help='checkpoint path', default='', type=str)
parser.add_argument('--save', default='')

args = parser.parse_args()
os.makedirs(args.save, exist_ok=True)
crop_size = args.crop_size
val_batch_size = args.val_batch_size
exp_name = args.exp_name

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

val_data_dir = args.val_root

# --- Gpu device --- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = Restorer()

net = net.to(device)

# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:
    os.mkdir('./{}/'.format(exp_name))

net.load_state_dict(torch.load(args.ckpt_path),strict=True)


val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False)
net.eval()

psnr, ssim = validation(net, val_data_loader, device, args.save, task=args.task)
print('psnr: {0:.2f}, ssim: {1:.4f}'.format(psnr, ssim))
