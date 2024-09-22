import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
import cv2
import os
import skimage
from PIL import Image
import cv2
from skimage.measure import compare_psnr, compare_ssim
import pdb
from txt_utils import clip
from torchvision.transforms import Compose, ToTensor, Normalize

def calc_psnr(im1, im2):

    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()


    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y)]
    return ans

def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]
    return ans

def to_psnr(pred_image, gt):
    mse = F.mse_loss(pred_image, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list


def validation(net, val_data_loader, device, save_path, task='blur', save_tag=True):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt, gt_path = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            text = clip.tokenize(task).to(device)
            pred_image = net(input_im, text)

# --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, gt_path, save_path)

    print(len(psnr_list))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def validation_val(net, val_data_loader, device, save_path, save_tag=False):

    psnr_list = []
    ssim_list = []

    for batch_id, val_data, gt_path in enumerate(val_data_loader):

        with torch.no_grad():
            input_im, gt = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image, _ = net(input_im)

# --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, gt_path, save_path)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def save_image(pred_image, gt_path, save_path):
    # pred_image_images = torch.split(pred_image, 1, dim=0)
    # batch_num = len(pred_image_images)
    #
    # for ind in range(batch_num):
    #     image_name_1 = image_name[ind].split('/')[-1]
    #     print(image_name_1)
    print(gt_path)
    image_name = os.path.basename(gt_path[0])
    print(pred_image.shape)
    utils.save_image(pred_image, os.path.join(save_path, image_name))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format( exp_name), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)



def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 50
    # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    elif epoch > 60:
        if not epoch % step and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
                print('Learning rate sets to {}.'.format(param_group['lr']))
        else:
            for param_group in optimizer.param_groups:
                print('Learning rate sets to {}.'.format(param_group['lr']))


def test(device, img_root, gt_root):

    psnr_list = []
    ssim_list = []

    paths = os.listdir(gt_root)
    transform_gt = Compose([ToTensor()])

    for batch_id, path in enumerate(paths):

        with torch.no_grad():

            pred_path = os.path.join(img_root, path)
            print(pred_path)
            gt_path = os.path.join(gt_root, path)

            pred_image = Image.open(pred_path).convert("RGB").resize((256, 256), Image.ANTIALIAS)
            gt = Image.open(gt_path).convert("RGB").resize((256, 256), Image.ANTIALIAS)
            pred_image = transform_gt(pred_image)
            gt = transform_gt(gt)


            pred_image = pred_image.to(device).unsqueeze(dim=0)
            gt = gt.to(device).unsqueeze(dim=0)

# --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(pred_image, gt))


    print(len(psnr_list))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim
# from collections import OrderedDict
# ckpt = torch.load(r'/home/cyq/Desktop/Codes/transweather/None/latest1')
# new_state_dict = OrderedDict()
#
# for k,v in ckpt.items():
#     name = k[7:]
#     new_state_dict[name] = v
#RealBlur_R
# torch.save(new_state_dict,'./latest.pth')

# img_root = r'E:\DEALL\result\OneRestore\haze_snow'
# gt_root = r'E:\DEALL\CDD-11_test\clear'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# old_val_psnr1, old_val_ssim1 = test(device, img_root, gt_root)
# print('CSD old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))
