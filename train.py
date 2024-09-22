import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random
from models.restorer import Restorer
from txt_utils import clip
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1.25e-5, type=float)
parser.add_argument('-train_root', help='train dataset path', default='', type=str)
parser.add_argument('-val_root', help='test dataset path', default='', type=str)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=24, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=300, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs


#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


train_data_dir = args.train_root
val_data_dir = args.val_root

# --- Gpu device --- #
device_ids = [0,1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Restorer()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))
try:
    net.module.load_state_dict(torch.load('./{}/latest.pth'.format(exp_name)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(train_data_dir, crop_size), batch_size=train_batch_size, shuffle=True, num_workers=8,drop_last=True)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Previous PSNR and SSIM in testing --- #
net.eval()

################ Note########################
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, exp_name)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

net.train()

for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, target = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)
        target = clip.tokenize(target).to(device)
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image, target)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        loss = smooth_loss + lambda_loss*perceptual_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # scheduler_cosine.step()
    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.module.state_dict(), './{}/latest.pth'.format(exp_name))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name)

    torch.save(net.module.state_dict(), f'./{exp_name}/{epoch}_{val_psnr}.pth')
    print('model saved')

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.module.state_dict(), './{}/best.pth'.format(exp_name))
        print('model saved')
        old_val_psnr = val_psnr

