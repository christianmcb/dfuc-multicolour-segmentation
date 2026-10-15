import os
import skimage
import imageio as io
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np

from glob import glob
from skimage.io import imsave, imread
import torch

# apply upsampling to images

src_images = glob('<DATA_ROOT>/inputs/dfuc2022/train/images/*')
output_image_path = '<DATA_ROOT>/inputs/dfuc2022_cutmix/train/images/'
output_mask_path = '<DATA_ROOT>/inputs/dfuc2022_cutmix/train/masks/'

def CutMix(data, targets, alpha):
	indices = torch.randperm(data.size(0))
	shuffled_data = data[indices]
	shuffled_targets = targets[indices]
	
	lam = np.random.beta(alpha, alpha)
	bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
	data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
	# adjust lambda to exactly match pixel ratio
	lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
	
	new_targets = [targets, shuffled_targets, lam]
	return data, new_targets
	
def rand_bbox(size, lam):
	W = size[2]
	H = size[3]
	cut_rat = np.sqrt(1. - lam)
	cut_w = int(W * cut_rat)
	cut_h = int(H * cut_rat)
	
	# uniform
	cx = np.random.randint(W)
	cy = np.random.randint(H)
	
	bbx1 = np.clip(cx - cut_w // 2, 0, W)
	bby1 = np.clip(cy - cut_h // 2, 0, H)
	bbx2 = np.clip(cx + cut_w // 2, 0, W)
	bby2 = np.clip(cy + cut_h // 2, 0, H)
	
	return bbx1, bby1, bbx2, bby2

if not os.path.isdir(output_image_path):
  os.mkdir(output_image_path)
if not os.path.isdir(output_mask_path):
  os.mkdir(output_mask_path)
  
for i in range(0, len(src_images), 2):
  img_path, img_path2 = src_images[i], src_images[i+1]
  drive, path_and_file = os.path.splitdrive(img_path)
  path, file = os.path.split(path_and_file)
  
  img = torch.tensor(imread(img_path)).unsqueeze(0)
  img2 = torch.tensor(imread(img_path2)).unsqueeze(0)
  
  img, img2 = CutMix(img, img2, alpha=0.8)
  img, img2 = np.array(img[0]), np.array(img2[0])
  
  print(img2[0])
  print(img2[0].shape)
  
  print('Image path: ', output_image_path + file)
  #imsave(output_image_path + 'cutmix_' + file, img)
  imsave(output_image_path + 'cutmix_' + file, img2[0])
